//! NNUE evaluation: a small learned network that can replace the hand-written eval.
//!
//! Layout: 768 inputs (piece type x colour x square, seen from one side) feed a hidden
//! layer of `HIDDEN` neurons. It is computed twice, once from the side to move's view and
//! once from the opponent's, then both go through a clipped ReLU into one output neuron.

use std::fs;
use std::path::Path;
use std::sync::{Arc, RwLock};

use shakmaty::{Board, Color, Piece, Square};

pub const INPUTS: usize = 768;
pub const HIDDEN: usize = 128;
/// Output 1.0 means `SCALE` centipawns.
pub const SCALE: i32 = 400;
/// Quantisation factors: hidden layer weights are stored times QA, output weights times QB.
pub const QA: i32 = 255;
pub const QB: i32 = 64;
/// Float weights are kept inside this range while training so they fit in i16 after scaling.
pub const WEIGHT_CLIP: f32 = 1.98;

const MAGIC: &[u8; 4] = b"ILN1";
const HEADER_BYTES: usize = 8;

/// Input index of `piece` on `square`, seen from `perspective`'s side of the board.
pub fn feature(perspective: Color, piece: Piece, square: Square) -> usize {
    let square = square as usize;
    let relative_square = if perspective.is_white() { square } else { square ^ 56 };
    let theirs = usize::from(piece.color != perspective);
    theirs * 384 + (piece.role as usize - 1) * 64 + relative_square
}

/// Active inputs for both views: (side to move, other side). At most 32 each.
pub fn active_features(board: &Board, side_to_move: Color) -> (Vec<usize>, Vec<usize>) {
    let mut ours = Vec::with_capacity(32);
    let mut theirs = Vec::with_capacity(32);
    for square in board.occupied() {
        if let Some(piece) = board.piece_at(square) {
            ours.push(feature(side_to_move, piece, square));
            theirs.push(feature(!side_to_move, piece, square));
        }
    }
    (ours, theirs)
}

/// Quantised network used during search.
#[derive(Clone, Debug, PartialEq)]
pub struct Network {
    pub input_weights: Vec<i16>, // INPUTS x HIDDEN, row per input
    pub input_bias: Vec<i16>,    // HIDDEN
    pub output_weights: Vec<i16>, // 2 x HIDDEN: side to move first
    pub output_bias: i32,
}

impl Network {
    pub fn load(path: &Path) -> Result<Network, String> {
        let bytes = fs::read(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
        Network::from_bytes(&bytes)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Network, String> {
        let expected = HEADER_BYTES + 2 * (INPUTS * HIDDEN + HIDDEN + 2 * HIDDEN) + 4;
        if bytes.len() != expected || &bytes[..4] != MAGIC {
            return Err("not an Illusion network file (or made for a different size)".into());
        }
        let hidden = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
        if hidden != HIDDEN {
            return Err(format!("network has {hidden} hidden neurons, engine expects {HIDDEN}"));
        }
        let mut words = bytes[HEADER_BYTES..].chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]]));
        let mut take = |n: usize| words.by_ref().take(n).collect::<Vec<i16>>();
        let input_weights = take(INPUTS * HIDDEN);
        let input_bias = take(HIDDEN);
        let output_weights = take(2 * HIDDEN);
        let tail = &bytes[bytes.len() - 4..];
        let output_bias = i32::from_le_bytes(tail.try_into().unwrap());
        Ok(Network { input_weights, input_bias, output_weights, output_bias })
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(HEADER_BYTES + 2 * (INPUTS + 3) * HIDDEN + 4);
        bytes.extend_from_slice(MAGIC);
        bytes.extend_from_slice(&(HIDDEN as u32).to_le_bytes());
        for w in self.input_weights.iter().chain(&self.input_bias).chain(&self.output_weights) {
            bytes.extend_from_slice(&w.to_le_bytes());
        }
        bytes.extend_from_slice(&self.output_bias.to_le_bytes());
        bytes
    }

    /// Centipawns from the side to move's view.
    pub fn evaluate(&self, board: &Board, side_to_move: Color) -> i32 {
        let mut accumulators = [[0i16; HIDDEN]; 2];
        accumulators[0].copy_from_slice(&self.input_bias);
        accumulators[1].copy_from_slice(&self.input_bias);
        for square in board.occupied() {
            let Some(piece) = board.piece_at(square) else { continue };
            for (view, perspective) in [side_to_move, !side_to_move].into_iter().enumerate() {
                let row = feature(perspective, piece, square) * HIDDEN;
                let weights = &self.input_weights[row..row + HIDDEN];
                for (acc, w) in accumulators[view].iter_mut().zip(weights) {
                    *acc = acc.wrapping_add(*w);
                }
            }
        }
        let mut sum = 0i32;
        for (view, accumulator) in accumulators.iter().enumerate() {
            let weights = &self.output_weights[view * HIDDEN..(view + 1) * HIDDEN];
            for (&acc, &w) in accumulator.iter().zip(weights) {
                sum += i32::from(acc.clamp(0, QA as i16)) * i32::from(w);
            }
        }
        (sum + self.output_bias) * SCALE / (QA * QB)
    }
}

static ACTIVE: RwLock<Option<Arc<Network>>> = RwLock::new(None);

/// Use `network` for evaluation from now on (`None` goes back to the hand-written eval).
pub fn set_active(network: Option<Network>) {
    *ACTIVE.write().unwrap() = network.map(Arc::new);
}

pub fn active() -> Option<Arc<Network>> {
    ACTIVE.read().unwrap().clone()
}
