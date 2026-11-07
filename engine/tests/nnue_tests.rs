mod common;

use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use common::{pos, START};
use illusion_engine::datagen::{play_game, Outcome, Rng, Sample, SAMPLE_BYTES};
use illusion_engine::nnue::{self, Network, HIDDEN, INPUTS, QA, QB};
use illusion_engine::search::Searcher;
use illusion_engine::train::{float_eval, initial_params, quantize, train, TrainConfig};
use shakmaty::{Color, Piece, Position, Role, Square};

const MIDDLEGAME: &str = "r1b2rk1/pp1nqppp/2pbpn2/3p4/2PP4/2NBPN2/PPQ2PPP/R1B2RK1 w - - 4 9";

fn sample(fen: &str, score: i32, outcome: Outcome) -> Sample {
    let position = pos(fen);
    Sample::new(position.board(), position.turn(), score, outcome)
}

#[test]
fn features_cover_every_input_exactly_once() {
    let mut seen = vec![false; INPUTS];
    for perspective in [Color::White, Color::Black] {
        seen.fill(false);
        for color in [Color::White, Color::Black] {
            for role in Role::ALL {
                for square in Square::ALL {
                    let index = nnue::feature(perspective, Piece { color, role }, square);
                    assert!(!seen[index], "duplicate input {index}");
                    seen[index] = true;
                }
            }
        }
        assert!(seen.iter().all(|&s| s));
    }
}

#[test]
fn both_sides_see_their_own_pieces_the_same_way() {
    let white_pawn_e2 = nnue::feature(Color::White, Piece { color: Color::White, role: Role::Pawn }, Square::E2);
    let black_pawn_e7 = nnue::feature(Color::Black, Piece { color: Color::Black, role: Role::Pawn }, Square::E7);
    assert_eq!(white_pawn_e2, black_pawn_e7);
}

#[test]
fn samples_survive_a_round_trip_to_bytes() {
    let original = sample(MIDDLEGAME, -37, Outcome::BlackWins);
    let bytes = original.to_bytes();
    assert_eq!(bytes.len(), SAMPLE_BYTES);
    let back = Sample::from_bytes(&bytes).unwrap();
    assert_eq!(back, original);
    let position = pos(MIDDLEGAME);
    let pieces: Vec<_> = back.pieces().collect();
    assert_eq!(pieces.len(), position.board().occupied().count());
    for (square, piece) in pieces {
        assert_eq!(position.board().piece_at(square), Some(piece));
    }
}

#[test]
fn corrupt_samples_are_rejected() {
    let mut bytes = sample(START, 0, Outcome::Draw).to_bytes();
    bytes[26] = 7;
    assert!(Sample::from_bytes(&bytes).is_err());
    assert!(Sample::from_bytes(&bytes[..10]).is_err());
}

#[test]
fn outcome_scores_are_from_the_right_side() {
    assert_eq!(Outcome::WhiteWins.score_for(Color::White), 1.0);
    assert_eq!(Outcome::WhiteWins.score_for(Color::Black), 0.0);
    assert_eq!(Outcome::Draw.score_for(Color::Black), 0.5);
}

#[test]
fn network_files_survive_a_round_trip() {
    let network = quantize(&initial_params(7));
    let back = Network::from_bytes(&network.to_bytes()).unwrap();
    assert_eq!(back, network);
    assert_eq!(back.input_weights.len(), INPUTS * HIDDEN);
    assert!(Network::from_bytes(b"not a network").is_err());
}

#[test]
fn integer_network_matches_the_float_one() {
    // snap the float weights onto the integer grid so only real maths differences remain
    let network = quantize(&initial_params(11));
    let mut params: Vec<f32> = network.input_weights.iter().chain(&network.input_bias).map(|&w| f32::from(w) / QA as f32).collect();
    params.extend(network.output_weights.iter().map(|&w| f32::from(w) / QB as f32));
    params.push(network.output_bias as f32 / (QA * QB) as f32);
    for fen in [START, MIDDLEGAME, "8/5k2/8/3P4/8/8/2K5/8 b - - 0 1"] {
        let position = pos(fen);
        let exact = float_eval(&params, &sample(fen, 0, Outcome::Draw));
        let quantized = network.evaluate(position.board(), position.turn()) as f32;
        assert!((exact - quantized).abs() <= 1.5, "{fen}: float {exact} vs int {quantized}");
    }
}

#[test]
fn self_play_records_quiet_positions_with_a_result() {
    let mut searcher = Searcher::new(4, Arc::new(AtomicBool::new(false)));
    let samples = play_game(&mut searcher, &mut Rng::new(3), 300);
    assert!(!samples.is_empty());
    let outcome = samples[0].outcome;
    assert!(samples.iter().all(|s| s.outcome == outcome));
    assert!(samples.iter().all(|s| i32::from(s.score).abs() <= 3000));
}

#[test]
fn training_lowers_the_loss_on_a_simple_pattern() {
    // white is better exactly when it has an extra queen
    let mut samples = Vec::new();
    for i in 0..400 {
        let (fen, score, outcome) = if i % 2 == 0 {
            ("4k3/8/8/8/8/8/8/3QK3 w - - 0 1", 900, Outcome::WhiteWins)
        } else {
            ("3qk3/8/8/8/8/8/8/4K3 w - - 0 1", -900, Outcome::BlackWins)
        };
        samples.push(sample(fen, score, outcome));
    }
    let out = std::env::temp_dir().join(format!("illusion-test-{}.nnue", std::process::id()));
    let config = TrainConfig { epochs: 30, batch_size: 64, learning_rate: 0.01, threads: 2, ..TrainConfig::default() };
    let (params, history) = train(samples, &out, &config).unwrap();
    assert!(history.last().unwrap() < &history[0], "{history:?}");

    let network = Network::load(&out).unwrap();
    std::fs::remove_file(&out).ok();
    let up = pos("4k3/8/8/8/8/8/8/3QK3 w - - 0 1");
    let down = pos("3qk3/8/8/8/8/8/8/4K3 w - - 0 1");
    assert!(network.evaluate(up.board(), up.turn()) > 200);
    assert!(network.evaluate(down.board(), down.turn()) < -200);
    let up_sample = sample("4k3/8/8/8/8/8/8/3QK3 w - - 0 1", 0, Outcome::Draw);
    assert!(float_eval(&params, &up_sample) > 200.0);
}
