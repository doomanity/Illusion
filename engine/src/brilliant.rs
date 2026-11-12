//! Brilliant hunter: look for a sound sacrifice that is (almost) as good as the best move
//! and play it instead. Every sacrifice is checked, then the promising ones are verified
//! with a deep search, so only sacrifices that hold up against best defence are played.

use std::time::Instant;

use shakmaty::{Chess, Move, Position};

use crate::eval::MATE_BOUND;
use crate::search::{position_key, Limits, SearchResult, Searcher};
use crate::see::see;
use crate::swindle::MAX_VERIFY_DEPTH;

pub const DEFAULT_HUNT: i32 = 30;
pub const MAX_HUNT: i32 = 100;

/// A move that gives up at least this much material by exchange is a sacrifice.
pub const SACRIFICE_MIN_LOSS: i32 = 100;
const QUICK_NODES: u64 = 20_000;
const VERIFY_AT_MOST: usize = 3;
/// Quick checks are rough, so let sacrifices this much below the limit through to verification.
const QUICK_SLACK: i32 = 80;

pub fn is_sacrifice(pos: &Chess, m: Move) -> bool {
    see(pos.board(), m) <= -SACRIFICE_MIN_LOSS
}

/// Score of `pos` for the side to move, searched quietly within a node budget, and the
/// depth that was actually completed.
fn score(searcher: &mut Searcher, pos: &Chess, history: &[u64], depth: i32, nodes: Option<u64>, deadline: Instant) -> (i32, i32) {
    let time_left = deadline.saturating_duration_since(Instant::now());
    let limits = Limits { max_depth: Some(depth), max_nodes: nodes, hard_time: Some(time_left), ..Limits::default() };
    let result = searcher.search(pos, history, limits);
    (result.score, result.depth)
}

/// Good enough to play instead of `best_score`: within `max_loss`, or mating either way.
fn acceptable(sacrifice_score: i32, best_score: i32, max_loss: i32) -> bool {
    let both_mate = sacrifice_score >= MATE_BOUND && best_score >= MATE_BOUND;
    both_mate || sacrifice_score >= best_score - max_loss
}

/// A sound sacrifice to play instead of the search's move, with its verified score.
pub fn hunt(
    searcher: &mut Searcher,
    pos: &Chess,
    history: &[u64],
    main: &SearchResult,
    max_loss: i32,
    deadline: Instant,
) -> Option<(Move, i32)> {
    let best = main.best_move?;
    if max_loss < 0 || main.score <= -MATE_BOUND || is_sacrifice(pos, best) {
        return None; // off, already lost, or the best move is a sacrifice anyway
    }
    let history: Vec<u64> = history.iter().copied().chain([position_key(pos)]).collect();
    let quick_depth = (main.depth - 4).clamp(2, 8);
    let out_of_time = |searcher: &Searcher| Instant::now() >= deadline || searcher.stopped();

    let mut promising = Vec::new();
    for m in pos.legal_moves() {
        if !is_sacrifice(pos, m) {
            continue;
        }
        if out_of_time(searcher) {
            return None;
        }
        let child = pos.clone().play(m).expect("legal move");
        let quick = -score(searcher, &child, &history, quick_depth, Some(QUICK_NODES), deadline).0;
        if acceptable(quick, main.score, max_loss + QUICK_SLACK) {
            promising.push((m, quick));
        }
    }
    promising.sort_by_key(|&(_, quick)| -quick);

    let verify_depth = (main.depth - 1).clamp(4, MAX_VERIFY_DEPTH); // as deep as the main search looked
    let mut found: Option<(Move, i32)> = None;
    for (m, _) in promising.into_iter().take(VERIFY_AT_MOST) {
        if out_of_time(searcher) {
            break;
        }
        let child = pos.clone().play(m).expect("legal move");
        let (child_score, completed) = score(searcher, &child, &history, verify_depth, None, deadline);
        if completed < verify_depth && child_score.abs() < MATE_BOUND {
            continue; // ran out of budget: a half-checked sacrifice is not played
        }
        let verified = -child_score;
        if acceptable(verified, main.score, max_loss) && found.is_none_or(|(_, s)| verified > s) {
            found = Some((m, verified));
        }
    }
    found
}
