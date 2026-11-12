//! Trap mode: among moves that are nearly as good as the best one, play the one that
//! scores best against a human of a given rating instead of against perfect defence.

use std::time::Instant;

use shakmaty::{Chess, Move, Position};

use crate::eval::{MATE, MATE_BOUND};
use crate::search::{position_key, Limits, SearchResult, Searcher};
use crate::see::see;

pub const DEFAULT_OPPONENT_ELO: i32 = 1500;
pub const DEFAULT_TOLERANCE: i32 = 60;

const SCORE_CLIP: i32 = 2000; // stops mate scores from wrecking the averages
const ALREADY_WINNING: i32 = 400; // just convert, no need for tricks
const MAX_CANDIDATES: usize = 5;
const MIN_GAIN: f64 = 10.0; // a trap must beat the best move's expectation by this much
const ROOT_NODES: u64 = 10_000;
/// Verification searches go this deep at most, so they can finish within a move's time.
pub const MAX_VERIFY_DEPTH: i32 = 12;
/// Against opponents this strong traps are not worth anything: no move may lose eval for one.
const NO_TRAPS_FROM_ELO: i32 = 2600;
/// Elo range over which the allowed trap cost grows from nothing to the full tolerance.
const TOLERANCE_RAMP_ELO: i32 = 800;
const REPLY_NODES: u64 = 4_000;

const MIN_ELO_FOR_NOISE: f64 = 600.0;
const MAX_ELO_FOR_NOISE: f64 = 2400.0;
const MIN_TEMPERATURE: f64 = 20.0;
const MAX_TEMPERATURE: f64 = 400.0;
const LOGIT_FLOOR: f64 = -50.0;

const CAPTURE_BONUS: f64 = 1.0;
const CHECK_BONUS: f64 = 0.7;
const PROMOTION_BONUS: f64 = 1.0;
const RECAPTURE_BONUS: f64 = 1.0;

const BRILLIANT_MIN_SACRIFICE: i32 = 200;
/// A move that gives up at least this much material (by exchange count) is a sacrifice.
const SACRIFICE_MIN_LOSS: i32 = 100;
/// Extra expected centipawns a sacrifice gets when sacrifice hunting is on.
pub const SACRIFICE_HUNT_BONUS: i32 = 40;
/// Anglerfish: a reply counts as swallowing the bait when it looks at least this much
/// better to the opponent than it really is.
const BAIT_MIN_ILLUSION: i32 = 100;
/// ... and when the opponent plays it at least this often.
const BAIT_MIN_CHANCE: f64 = 0.1;
const PERCEPTION_NODES: u64 = 1_500;
const BRILLIANT_MAX_BEFORE: i32 = 300;

#[derive(Clone, Copy, Debug)]
pub struct Settings {
    pub opponent_elo: i32,
    pub tolerance: i32,
    /// Bonus for sound sacrifices when choosing among safe moves (0 = off).
    pub sacrifice_bonus: i32,
    /// Model the opponent as judging replies by how they look a few moves deep (their
    /// elo decides how deep) instead of by their true value, so bait gets set.
    pub anglerfish: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Settings { opponent_elo: DEFAULT_OPPONENT_ELO, tolerance: DEFAULT_TOLERANCE, sacrifice_bonus: 0, anglerfish: false }
    }
}

#[derive(Clone, Debug)]
pub struct Choice {
    pub mv: Move,
    pub expected: f64,
    pub best_expected: f64,
    pub brilliant: bool,
    /// The opponent reply this move baits: it looks good to them but loses.
    pub bait: Option<Move>,
}

/// How a move is expected to go against the human model.
struct Outlook {
    expected: f64,
    worst: i32,
    bait: Option<Move>,
}

/// How many plies deep a player of this rating sees when judging a move.
pub fn perception_depth(elo: i32) -> i32 {
    match elo {
        ..1200 => 1,
        1200..1700 => 2,
        1700..2200 => 3,
        _ => 4,
    }
}

/// How many centipawns a trap may cost against this opponent: the full tolerance against
/// players 800 elo below `NO_TRAPS_FROM_ELO` or weaker, nothing against strong ones.
pub fn effective_tolerance(settings: Settings) -> i32 {
    let gap = (NO_TRAPS_FROM_ELO - settings.opponent_elo).clamp(0, TOLERANCE_RAMP_ELO);
    settings.tolerance * gap / TOLERANCE_RAMP_ELO
}

/// How much a player's choice blurs between moves, in centipawns. Higher means weaker.
pub fn temperature(elo: i32) -> f64 {
    let weakness = ((MAX_ELO_FOR_NOISE - elo as f64) / (MAX_ELO_FOR_NOISE - MIN_ELO_FOR_NOISE)).clamp(0.0, 1.0);
    MIN_TEMPERATURE + (MAX_TEMPERATURE - MIN_TEMPERATURE) * weakness
}

/// How strongly a player is pulled towards captures, checks and recaptures.
pub fn impulsiveness(elo: i32) -> f64 {
    (1.5 - elo as f64 / MAX_ELO_FOR_NOISE).max(0.0)
}

/// Chance of each reply. `quality` is the reply's score for the player making it,
/// `instinct` how tempting it looks (captures, checks...).
pub fn reply_probabilities(quality: &[f64], instinct: &[f64], elo: i32) -> Vec<f64> {
    let (temp, impulse) = (temperature(elo), impulsiveness(elo));
    let logits: Vec<f64> =
        quality.iter().zip(instinct).map(|(q, i)| (q / temp).max(LOGIT_FLOOR) + i * impulse).collect();
    let peak = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let weights: Vec<f64> = logits.iter().map(|l| (l - peak).exp()).collect();
    let total: f64 = weights.iter().sum();
    weights.iter().map(|w| w / total).collect()
}

fn instinct(pos: &Chess, reply: Move, previous: Move) -> f64 {
    let mut bonus = 0.0;
    if reply.is_capture() {
        bonus += CAPTURE_BONUS;
    }
    if reply.is_promotion() {
        bonus += PROMOTION_BONUS;
    }
    if reply.to() == previous.to() {
        bonus += RECAPTURE_BONUS;
    }
    if pos.clone().play(reply).is_ok_and(|after| after.is_check()) {
        bonus += CHECK_BONUS;
    }
    bonus
}

fn clip(score: i32) -> f64 {
    score.clamp(-SCORE_CLIP, SCORE_CLIP) as f64
}

/// Quiet, budgeted search of `pos`. Score is for the side to move there.
fn probe(searcher: &mut Searcher, pos: &Chess, history: &[u64], depth: i32, nodes: u64) -> i32 {
    if pos.is_checkmate() {
        return -MATE;
    }
    let limits = Limits { max_depth: Some(depth), max_nodes: Some(nodes), ..Limits::default() };
    searcher.search(pos, history, limits).score
}

struct Explorer<'a> {
    searcher: &'a mut Searcher,
    history: Vec<u64>,
    deadline: Instant,
    reply_depth: i32,
}

impl Explorer<'_> {
    /// Deeper check that a trap move really keeps us above `floor` against best defence.
    fn holds_up(&mut self, pos: &Chess, mv: Move, floor: i32, main_depth: i32) -> bool {
        let child = pos.clone().play(mv).expect("legal move");
        let depth = (main_depth - 2).clamp(4, MAX_VERIFY_DEPTH);
        let time_left = self.deadline.saturating_duration_since(Instant::now());
        let limits = Limits { max_depth: Some(depth), hard_time: Some(time_left), ..Limits::default() };
        let result = self.searcher.search(&child, &self.history, limits);
        (result.depth >= depth || result.score.abs() >= MATE_BOUND) && -result.score >= floor
    }

    fn out_of_time(&self) -> bool {
        Instant::now() >= self.deadline || self.searcher.stopped()
    }

    /// Our expected score after `mv` against the human model, and the worst case.
    /// Our expected score after `mv` against the human model, the worst case, and the
    /// reply it baits (if any).
    fn expectation(&mut self, child: &Chess, mv: Move, settings: Settings) -> Option<Outlook> {
        let replies = child.legal_moves();
        if replies.is_empty() {
            let score = if child.is_check() { MATE } else { 0 };
            return Some(Outlook { expected: clip(score), worst: score, bait: None });
        }
        self.history.push(position_key(child));
        let mut ours = Vec::with_capacity(replies.len());
        let mut seen = Vec::with_capacity(replies.len()); // our score as the opponent perceives it
        let mut instincts = Vec::with_capacity(replies.len());
        let perception = perception_depth(settings.opponent_elo);
        for &reply in &replies {
            if self.out_of_time() {
                self.history.pop();
                return None;
            }
            let after = child.clone().play(reply).expect("legal reply");
            let truth = probe(self.searcher, &after, &self.history, self.reply_depth, REPLY_NODES);
            ours.push(truth);
            seen.push(if settings.anglerfish {
                probe(self.searcher, &after, &self.history, perception, PERCEPTION_NODES)
            } else {
                truth
            });
            instincts.push(instinct(child, reply, mv));
        }
        self.history.pop();
        let quality: Vec<f64> = seen.iter().map(|&s| -s as f64).collect();
        let probabilities = reply_probabilities(&quality, &instincts, settings.opponent_elo);
        let expected = probabilities.iter().zip(&ours).map(|(p, &s)| p * clip(s)).sum();

        // the bait: a likely reply that looks much better to them than it really is
        let bait = replies
            .iter()
            .zip(probabilities.iter().zip(ours.iter().zip(&seen)))
            .filter(|(_, (&p, (&truth, &looks)))| p >= BAIT_MIN_CHANCE && truth - looks >= BAIT_MIN_ILLUSION)
            .max_by(|a, b| {
                let gain = |(_, (&p, (&truth, &looks))): &(&Move, (&f64, (&i32, &i32)))| p * f64::from(truth - looks);
                gain(a).total_cmp(&gain(b))
            })
            .map(|(&reply, _)| reply);
        Some(Outlook { expected, worst: *ours.iter().min().unwrap(), bait })
    }
}

/// Look for a trap move after a normal search. `None` keeps the search's own move.
pub fn choose(
    searcher: &mut Searcher,
    pos: &Chess,
    history: &[u64],
    main: &SearchResult,
    settings: Settings,
    deadline: Instant,
) -> Option<Choice> {
    let best = main.best_move?;
    if main.score.abs() >= MATE_BOUND || main.score >= ALREADY_WINNING {
        return None;
    }
    let root_depth = (main.depth - 3).clamp(2, 8);
    let mut explorer = Explorer {
        searcher,
        history: history.iter().copied().chain([position_key(pos)]).collect(),
        deadline,
        reply_depth: (root_depth - 2).max(1),
    };

    let tolerance = effective_tolerance(settings);
    if tolerance == 0 {
        return None;
    }
    let floor = main.score - tolerance;
    let mut candidates = vec![(best, main.score)];
    for m in pos.legal_moves() {
        if m == best {
            continue;
        }
        if explorer.out_of_time() {
            break;
        }
        let child = pos.clone().play(m).expect("legal move");
        let score = -probe(explorer.searcher, &child, &explorer.history, root_depth, ROOT_NODES);
        if score >= floor {
            candidates.push((m, score));
        }
    }
    candidates[1..].sort_by_key(|&(_, score)| -score);
    candidates.truncate(MAX_CANDIDATES);
    if candidates.len() < 2 {
        return None;
    }

    // what each move is worth to us when choosing: expected score plus the sacrifice bonus
    let appeal = |m: Move, expected: f64| {
        let sacrifice = settings.sacrifice_bonus > 0 && see(pos.board(), m) <= -SACRIFICE_MIN_LOSS;
        expected + if sacrifice { f64::from(settings.sacrifice_bonus) } else { 0.0 }
    };
    let mut best_expected = None;
    let mut chosen: Option<(Move, f64, f64, Option<Move>)> = None; // move, expected, appeal, bait
    for (m, _) in candidates {
        let child = pos.clone().play(m).expect("legal move");
        let Some(outlook) = explorer.expectation(&child, m, settings) else { break };
        let value = appeal(m, outlook.expected);
        if m == best {
            best_expected = Some((outlook.expected, value));
            chosen = Some((m, outlook.expected, value, outlook.bait));
            continue;
        }
        let safe = outlook.worst >= floor;
        if safe && chosen.is_some_and(|(_, _, v, _)| value > v) {
            chosen = Some((m, outlook.expected, value, outlook.bait));
        }
    }

    let (best_expected, best_appeal) = best_expected?;
    let (mv, expected, appeal_value, bait) = chosen?;
    if mv != best && appeal_value < best_appeal + MIN_GAIN {
        return None;
    }
    if mv != best && !explorer.holds_up(pos, mv, floor, main.depth) {
        return None;
    }
    let sacrifice = see(pos.board(), mv) <= -BRILLIANT_MIN_SACRIFICE;
    let brilliant = sacrifice && main.score <= BRILLIANT_MAX_BEFORE;
    Some(Choice { mv, expected, best_expected, brilliant, bait })
}
