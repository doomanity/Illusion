//! Training data: the engine plays itself and records positions with their search score
//! and the final game result.
//!
//! Each position is stored in a fixed 32-byte record so files can be appended to, cut,
//! and joined freely.

use std::fs::OpenOptions;
use std::io::{self, BufWriter, Write};
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use shakmaty::{Bitboard, Board, Chess, Color, Piece, Position, Role, Square};

use crate::eval::MATE_BOUND;
use crate::nnue;
use crate::search::{position_key, Limits, Searcher};

pub const SAMPLE_BYTES: usize = 32;

const RANDOM_OPENING_PLIES: [usize; 2] = [8, 9];
const MAX_GAME_PLIES: usize = 400;
const MAX_RECORDED_SCORE: i32 = 3000;
const WIN_ADJUDICATION_SCORE: i32 = 2500;
const WIN_ADJUDICATION_PLIES: u32 = 4;
const DRAW_ADJUDICATION_FROM_PLY: usize = 80;
const DRAW_ADJUDICATION_SCORE: i32 = 10;
const DRAW_ADJUDICATION_PLIES: u32 = 10;
const TT_MB_PER_THREAD: usize = 16;
const PROGRESS_EVERY_GAMES: u64 = 200;

/// Game result from white's point of view.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    BlackWins = 0,
    Draw = 1,
    WhiteWins = 2,
}

impl Outcome {
    fn win_for(color: Color) -> Outcome {
        if color.is_white() { Outcome::WhiteWins } else { Outcome::BlackWins }
    }

    /// 1.0 win, 0.5 draw, 0.0 loss for `color`.
    pub fn score_for(self, color: Color) -> f32 {
        let white = self as u8 as f32 / 2.0;
        if color.is_white() { white } else { 1.0 - white }
    }
}

/// One recorded position.
#[derive(Clone, Debug, PartialEq)]
pub struct Sample {
    pub occupied: u64,
    pub pieces: [u8; 16], // 4 bits per piece in square order: colour << 3 | role
    pub score: i16,       // centipawns, white's view
    pub outcome: Outcome,
    pub side_to_move: Color,
}

impl Sample {
    pub fn new(board: &Board, side_to_move: Color, white_score: i32, outcome: Outcome) -> Sample {
        let mut pieces = [0u8; 16];
        for (i, square) in board.occupied().into_iter().enumerate() {
            let piece = board.piece_at(square).expect("occupied square");
            let code = (u8::from(piece.color.is_black()) << 3) | piece.role as u8;
            pieces[i / 2] |= code << (4 * (i % 2));
        }
        let score = white_score.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
        Sample { occupied: u64::from(board.occupied()), pieces, score, outcome, side_to_move }
    }

    /// Every (square, piece) on the board, in square order.
    pub fn pieces(&self) -> impl Iterator<Item = (Square, Piece)> + '_ {
        Bitboard(self.occupied).into_iter().enumerate().map(|(i, square)| {
            let code = (self.pieces[i / 2] >> (4 * (i % 2))) & 0xF;
            let color = if code & 8 == 0 { Color::White } else { Color::Black };
            let role = Role::try_from(code & 7).expect("valid role in sample");
            (square, Piece { color, role })
        })
    }

    pub fn to_bytes(&self) -> [u8; SAMPLE_BYTES] {
        let mut bytes = [0u8; SAMPLE_BYTES];
        bytes[..8].copy_from_slice(&self.occupied.to_le_bytes());
        bytes[8..24].copy_from_slice(&self.pieces);
        bytes[24..26].copy_from_slice(&self.score.to_le_bytes());
        bytes[26] = self.outcome as u8;
        bytes[27] = u8::from(self.side_to_move.is_black());
        bytes
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Sample, String> {
        if bytes.len() != SAMPLE_BYTES {
            return Err("sample record has the wrong size".into());
        }
        let outcome = match bytes[26] {
            0 => Outcome::BlackWins,
            1 => Outcome::Draw,
            2 => Outcome::WhiteWins,
            other => return Err(format!("bad outcome byte {other}")),
        };
        let occupied = u64::from_le_bytes(bytes[..8].try_into().unwrap());
        if occupied.count_ones() > 32 {
            return Err("too many pieces in sample".into());
        }
        let sample = Sample {
            occupied,
            pieces: bytes[8..24].try_into().unwrap(),
            score: i16::from_le_bytes([bytes[24], bytes[25]]),
            outcome,
            side_to_move: if bytes[27] == 0 { Color::White } else { Color::Black },
        };
        if sample.pieces().count() != occupied.count_ones() as usize {
            return Err("corrupt piece list in sample".into());
        }
        Ok(sample)
    }
}

/// Small fast random numbers (xorshift); fine for picking opening moves and shuffling.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed | 1)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

fn random_opening(rng: &mut Rng) -> Option<Chess> {
    let plies = RANDOM_OPENING_PLIES[rng.below(RANDOM_OPENING_PLIES.len())];
    let mut pos = Chess::default();
    for _ in 0..plies {
        let moves = pos.legal_moves();
        if moves.is_empty() {
            return None;
        }
        pos.play_unchecked(moves[rng.below(moves.len())]);
    }
    (!pos.legal_moves().is_empty()).then_some(pos)
}

/// Play one self-play game with `nodes` per move and return its recorded positions.
pub fn play_game(searcher: &mut Searcher, rng: &mut Rng, nodes: u64) -> Vec<Sample> {
    let pos = loop {
        if let Some(pos) = random_opening(rng) {
            break pos;
        }
    };
    searcher.new_game();
    let mut pos = pos;
    let mut history: Vec<u64> = Vec::new();
    let mut recorded: Vec<(Board, Color, i32)> = Vec::new();
    let (mut winning_plies, mut drawn_plies) = (0u32, 0u32);

    let outcome = loop {
        let key = position_key(&pos);
        let legal = pos.legal_moves();
        if legal.is_empty() {
            break if pos.is_check() { Outcome::win_for(!pos.turn()) } else { Outcome::Draw };
        }
        let repeated = history.iter().filter(|&&k| k == key).count() >= 2;
        if repeated || pos.halfmoves() >= 100 || pos.is_insufficient_material() || history.len() >= MAX_GAME_PLIES {
            break Outcome::Draw;
        }

        let result = searcher.search(&pos, &history, Limits::nodes(nodes));
        let Some(best) = result.best_move else { break Outcome::Draw };
        let score = result.score;
        if score.abs() >= MATE_BOUND {
            break if score > 0 { Outcome::win_for(pos.turn()) } else { Outcome::win_for(!pos.turn()) };
        }
        winning_plies = if score.abs() >= WIN_ADJUDICATION_SCORE { winning_plies + 1 } else { 0 };
        if winning_plies >= WIN_ADJUDICATION_PLIES {
            break if score > 0 { Outcome::win_for(pos.turn()) } else { Outcome::win_for(!pos.turn()) };
        }
        let quiet_draw = history.len() >= DRAW_ADJUDICATION_FROM_PLY && score.abs() <= DRAW_ADJUDICATION_SCORE;
        drawn_plies = if quiet_draw { drawn_plies + 1 } else { 0 };
        if drawn_plies >= DRAW_ADJUDICATION_PLIES {
            break Outcome::Draw;
        }

        let quiet = !pos.is_check() && !best.is_capture() && !best.is_promotion();
        if quiet && score.abs() <= MAX_RECORDED_SCORE {
            let white_score = if pos.turn().is_white() { score } else { -score };
            recorded.push((pos.board().clone(), pos.turn(), white_score));
        }
        history.push(key);
        pos.play_unchecked(best);
    };

    recorded.into_iter().map(|(board, turn, score)| Sample::new(&board, turn, score, outcome)).collect()
}

/// Settings for a data generation run.
#[derive(Clone, Copy, Debug)]
pub struct DatagenConfig {
    pub games: u64,
    pub threads: usize,
    pub nodes: u64,
}

/// Generate `config.games` games, appending positions to `out`. Safe to stop at any time:
/// every finished game is already on disk.
pub fn generate(out: &Path, config: DatagenConfig) -> io::Result<()> {
    let file = OpenOptions::new().create(true).append(true).open(out)?;
    let writer = Mutex::new(BufWriter::new(file));
    let started_games = AtomicU64::new(0);
    let finished_games = AtomicU64::new(0);
    let positions = AtomicU64::new(0);
    let failed = AtomicBool::new(false);
    let start = Instant::now();
    let seed = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(1, |d| d.as_nanos() as u64);
    let network = nnue::active();
    println!(
        "playing {} games on {} threads, {} nodes per move, eval: {}",
        config.games,
        config.threads,
        config.nodes,
        if network.is_some() { "network" } else { "hand-written" }
    );

    std::thread::scope(|scope| {
        for thread in 0..config.threads.max(1) {
            let (writer, started_games, finished_games, positions, failed) =
                (&writer, &started_games, &finished_games, &positions, &failed);
            scope.spawn(move || {
                let mut rng = Rng::new(seed ^ (thread as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15));
                let mut searcher = Searcher::new(TT_MB_PER_THREAD, Arc::new(AtomicBool::new(false)));
                while !failed.load(Ordering::Relaxed) && started_games.fetch_add(1, Ordering::Relaxed) < config.games {
                    let samples = play_game(&mut searcher, &mut rng, config.nodes);
                    let mut out = writer.lock().unwrap();
                    let written = samples.iter().try_for_each(|s| out.write_all(&s.to_bytes())).and_then(|_| out.flush());
                    if let Err(error) = written {
                        eprintln!("write failed: {error}");
                        failed.store(true, Ordering::Relaxed);
                        return;
                    }
                    drop(out);
                    let total = positions.fetch_add(samples.len() as u64, Ordering::Relaxed) + samples.len() as u64;
                    let done = finished_games.fetch_add(1, Ordering::Relaxed) + 1;
                    if done % PROGRESS_EVERY_GAMES == 0 {
                        let seconds = start.elapsed().as_secs_f64().max(0.001);
                        println!(
                            "{done} games, {total} positions, {:.0} positions/s",
                            total as f64 / seconds
                        );
                    }
                }
            });
        }
    });
    if failed.load(Ordering::Relaxed) {
        return Err(io::Error::other("writing training data failed"));
    }
    println!(
        "done: {} games, {} positions in {:.0}s",
        finished_games.load(Ordering::Relaxed),
        positions.load(Ordering::Relaxed),
        start.elapsed().as_secs_f64()
    );
    Ok(())
}
