//! Trainer for the NNUE network: learns from `datagen` files on the CPU.
//!
//! Each position is scored by `sigmoid(output)` and pulled towards a blend of the game
//! result and the search score it was recorded with. Weights are trained as floats and
//! saved quantised after every epoch, so a run can be stopped at any time.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::datagen::{Rng, Sample, SAMPLE_BYTES};
use crate::nnue::{self, Network, HIDDEN, INPUTS, QA, QB, SCALE, WEIGHT_CLIP};

const W1: usize = 0;
const B1: usize = W1 + INPUTS * HIDDEN;
const W2: usize = B1 + HIDDEN;
const B2: usize = W2 + 2 * HIDDEN;
pub const PARAMS: usize = B2 + 1;

const VALIDATION_EVERY: usize = 100; // one position in this many is held out
const BETA1: f32 = 0.9;
const BETA2: f32 = 0.999;
const EPSILON: f32 = 1e-8;
const INPUT_INIT_RANGE: f32 = 0.1;
const LR_FINAL_FRACTION: f32 = 0.1;

#[derive(Clone, Debug)]
pub struct TrainConfig {
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f32,
    /// How much the game result counts against the search score (0 = only score, 1 = only result).
    pub result_weight: f32,
    pub threads: usize,
    pub seed: u64,
}

impl Default for TrainConfig {
    fn default() -> Self {
        TrainConfig {
            epochs: 15,
            batch_size: 16_384,
            learning_rate: 0.001,
            result_weight: 0.3,
            threads: std::thread::available_parallelism().map_or(4, |n| n.get()),
            seed: 0x1111_5100,
        }
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Random starting weights.
pub fn initial_params(seed: u64) -> Vec<f32> {
    let mut rng = Rng::new(seed);
    let mut uniform = |range: f32| (rng.next_u64() as f64 / u64::MAX as f64) as f32 * 2.0 * range - range;
    let mut params = vec![0.0f32; PARAMS];
    params[W1..B1].iter_mut().for_each(|w| *w = uniform(INPUT_INIT_RANGE));
    let output_range = 1.0 / (2.0 * HIDDEN as f32).sqrt();
    params[W2..B2].iter_mut().for_each(|w| *w = uniform(output_range));
    params
}

/// Inputs seen from the side to move and from the other side.
fn features(sample: &Sample) -> ([usize; 32], [usize; 32], usize) {
    let (mut ours, mut theirs, mut count) = ([0usize; 32], [0usize; 32], 0);
    for (square, piece) in sample.pieces() {
        ours[count] = nnue::feature(sample.side_to_move, piece, square);
        theirs[count] = nnue::feature(!sample.side_to_move, piece, square);
        count += 1;
    }
    (ours, theirs, count)
}

/// What the network should output for this position, as a win probability.
fn target(sample: &Sample, result_weight: f32) -> f32 {
    let stm_score = if sample.side_to_move.is_white() { sample.score } else { -sample.score };
    let from_score = sigmoid(f32::from(stm_score) / SCALE as f32);
    result_weight * sample.outcome.score_for(sample.side_to_move) + (1.0 - result_weight) * from_score
}

/// Float forward pass: returns (hidden pre-activations for both views, output).
fn forward(params: &[f32], views: [&[usize]; 2]) -> ([[f32; HIDDEN]; 2], f32) {
    let mut accumulators = [[0f32; HIDDEN]; 2];
    let mut output = params[B2];
    for (view, inputs) in views.iter().enumerate() {
        let accumulator = &mut accumulators[view];
        accumulator.copy_from_slice(&params[B1..W2]);
        for &input in inputs.iter() {
            let row = &params[W1 + input * HIDDEN..W1 + (input + 1) * HIDDEN];
            accumulator.iter_mut().zip(row).for_each(|(a, w)| *a += w);
        }
        let weights = &params[W2 + view * HIDDEN..W2 + (view + 1) * HIDDEN];
        output += accumulator.iter().zip(weights).map(|(a, w)| a.clamp(0.0, 1.0) * w).sum::<f32>();
    }
    (accumulators, output)
}

/// Float network score in centipawns for the side to move (used to check quantisation).
pub fn float_eval(params: &[f32], sample: &Sample) -> f32 {
    let (ours, theirs, count) = features(sample);
    forward(params, [&ours[..count], &theirs[..count]]).1 * SCALE as f32
}

/// Add this batch's gradients into `grads` (when given) and return the summed loss.
fn batch_loss(params: &[f32], samples: &[Sample], result_weight: f32, mut grads: Option<&mut [f32]>) -> f64 {
    let mut loss = 0.0f64;
    for sample in samples {
        let (ours, theirs, count) = features(sample);
        let views = [&ours[..count], &theirs[..count]];
        let (accumulators, output) = forward(params, views);
        let prediction = sigmoid(output);
        let error = prediction - target(sample, result_weight);
        loss += f64::from(error * error);
        let Some(grads) = grads.as_deref_mut() else { continue };

        let g = 2.0 * error * prediction * (1.0 - prediction);
        grads[B2] += g;
        for (view, inputs) in views.iter().enumerate() {
            let mut hidden_grad = [0f32; HIDDEN];
            for i in 0..HIDDEN {
                let a = accumulators[view][i];
                grads[W2 + view * HIDDEN + i] += g * a.clamp(0.0, 1.0);
                if a > 0.0 && a < 1.0 {
                    hidden_grad[i] = g * params[W2 + view * HIDDEN + i];
                }
            }
            grads[B1..W2].iter_mut().zip(&hidden_grad).for_each(|(b, h)| *b += h);
            for &input in inputs.iter() {
                let row = &mut grads[W1 + input * HIDDEN..W1 + (input + 1) * HIDDEN];
                row.iter_mut().zip(&hidden_grad).for_each(|(w, h)| *w += h);
            }
        }
    }
    loss
}

/// Loss and gradient over `samples`, split across threads. Gradients are averaged.
fn parallel_loss(
    params: &[f32],
    samples: &[Sample],
    config: &TrainConfig,
    thread_grads: &mut [Vec<f32>],
    want_grads: bool,
) -> f64 {
    let chunk = samples.len().div_ceil(thread_grads.len()).max(1);
    let loss = std::thread::scope(|scope| {
        let handles: Vec<_> = samples
            .chunks(chunk)
            .zip(thread_grads.iter_mut())
            .map(|(part, grads)| {
                scope.spawn(move || {
                    if want_grads {
                        grads.fill(0.0);
                        batch_loss(params, part, config.result_weight, Some(grads))
                    } else {
                        batch_loss(params, part, config.result_weight, None)
                    }
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().expect("training thread panicked")).sum::<f64>()
    });
    if want_grads {
        let used = samples.len().div_ceil(chunk);
        let (first, rest) = thread_grads.split_at_mut(1);
        for other in &rest[..used - 1] {
            first[0].iter_mut().zip(other).for_each(|(a, b)| *a += b);
        }
        let scale = 1.0 / samples.len() as f32;
        first[0].iter_mut().for_each(|g| *g *= scale);
    }
    loss / samples.len().max(1) as f64
}

struct Adam {
    m: Vec<f32>,
    v: Vec<f32>,
    step: i32,
}

impl Adam {
    fn new() -> Adam {
        Adam { m: vec![0.0; PARAMS], v: vec![0.0; PARAMS], step: 0 }
    }

    fn update(&mut self, params: &mut [f32], grads: &[f32], learning_rate: f32) {
        self.step += 1;
        let correction1 = 1.0 - BETA1.powi(self.step);
        let correction2 = 1.0 - BETA2.powi(self.step);
        for i in 0..PARAMS {
            self.m[i] = BETA1 * self.m[i] + (1.0 - BETA1) * grads[i];
            self.v[i] = BETA2 * self.v[i] + (1.0 - BETA2) * grads[i] * grads[i];
            let m_hat = self.m[i] / correction1;
            let v_hat = self.v[i] / correction2;
            params[i] -= learning_rate * m_hat / (v_hat.sqrt() + EPSILON);
        }
        params[W1..B2].iter_mut().for_each(|w| *w = w.clamp(-WEIGHT_CLIP, WEIGHT_CLIP));
    }
}

/// Integer network for the engine.
pub fn quantize(params: &[f32]) -> Network {
    let scaled = |slice: &[f32], factor: i32| -> Vec<i16> {
        slice.iter().map(|w| (w * factor as f32).round() as i16).collect()
    };
    Network {
        input_weights: scaled(&params[W1..B1], QA),
        input_bias: scaled(&params[B1..W2], QA),
        output_weights: scaled(&params[W2..B2], QB),
        output_bias: (params[B2] * (QA * QB) as f32).round() as i32,
    }
}

/// Read every sample in the given `datagen` files.
pub fn load_samples(files: &[PathBuf]) -> Result<Vec<Sample>, String> {
    let mut samples = Vec::new();
    for file in files {
        let bytes = fs::read(file).map_err(|e| format!("cannot read {}: {e}", file.display()))?;
        if bytes.len() % SAMPLE_BYTES != 0 {
            eprintln!("{}: ignoring a cut-off record at the end", file.display());
        }
        for record in bytes.chunks_exact(SAMPLE_BYTES) {
            samples.push(Sample::from_bytes(record).map_err(|e| format!("{}: {e}", file.display()))?);
        }
    }
    Ok(samples)
}

/// Train on `samples`, writing the network to `out` after every epoch. Returns the final
/// float weights and the validation loss of each epoch.
pub fn train(samples: Vec<Sample>, out: &Path, config: &TrainConfig) -> io::Result<(Vec<f32>, Vec<f64>)> {
    let mut rng = Rng::new(config.seed);
    let mut samples = samples;
    shuffle(&mut samples, &mut rng);
    let validation_size = (samples.len() / VALIDATION_EVERY).max(1).min(samples.len());
    let validation = samples.split_off(samples.len() - validation_size);
    println!("training on {} positions, checking on {}", samples.len(), validation.len());

    let mut params = initial_params(config.seed);
    let mut adam = Adam::new();
    let mut thread_grads = vec![vec![0.0f32; PARAMS]; config.threads.max(1)];
    let mut history = Vec::with_capacity(config.epochs);

    for epoch in 0..config.epochs {
        let start = Instant::now();
        let progress = epoch as f32 / (config.epochs.max(2) - 1) as f32;
        let learning_rate = config.learning_rate * LR_FINAL_FRACTION.powf(progress);
        shuffle(&mut samples, &mut rng);
        let mut train_loss = 0.0;
        let mut batches = 0;
        for batch in samples.chunks(config.batch_size.max(1)) {
            train_loss += parallel_loss(&params, batch, config, &mut thread_grads, true);
            adam.update(&mut params, &thread_grads[0], learning_rate);
            batches += 1;
        }
        let validation_loss = parallel_loss(&params, &validation, config, &mut thread_grads, false);
        history.push(validation_loss);
        fs::write(out, quantize(&params).to_bytes())?;
        println!(
            "epoch {:>2}/{}  train loss {:.5}  check loss {:.5}  lr {:.6}  {:.0}s  saved {}",
            epoch + 1,
            config.epochs,
            train_loss / batches.max(1) as f64,
            validation_loss,
            learning_rate,
            start.elapsed().as_secs_f64(),
            out.display()
        );
    }
    Ok((params, history))
}

fn shuffle<T>(items: &mut [T], rng: &mut Rng) {
    for i in (1..items.len()).rev() {
        items.swap(i, rng.below(i + 1));
    }
}
