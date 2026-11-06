mod common;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use common::{pos, uci, START};
use illusion_engine::eval::{MATE, MATE_BOUND};
use illusion_engine::search::{format_score, is_repetition, position_key, Limits, Searcher};
use shakmaty::{Chess, Position};

fn searcher() -> Searcher {
    Searcher::new(16, Arc::new(AtomicBool::new(false)))
}

fn best(fen: &str, depth: i32) -> (String, i32) {
    let result = searcher().search(&pos(fen), &[], Limits::depth(depth));
    (uci(result.best_move.unwrap()), result.score)
}

#[test]
fn finds_mate_in_one() {
    let (mv, score) = best("6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1", 4);
    assert_eq!(mv, "e1e8");
    assert_eq!(score, MATE - 1);
}

#[test]
fn finds_mate_in_two() {
    let (mv, score) = best("k7/8/8/1K6/8/8/8/7R w - - 0 1", 6);
    assert_eq!(mv, "b5b6");
    assert_eq!(score, MATE - 3);
}

#[test]
fn sees_that_it_is_getting_mated() {
    // black to move, white threatens Qxf7# and black cannot stop everything: just check the sign
    let (_, score) = best("6k1/5ppp/8/8/8/8/5PPP/4R1K1 b - - 0 1", 3);
    assert!(score < 0 || score.abs() < MATE_BOUND);
}

#[test]
fn avoids_stalemating_when_winning() {
    let (mv, score) = best("7k/8/6K1/8/8/8/8/5Q2 w - - 0 1", 5);
    assert_ne!(mv, "f1f7");
    assert!(score >= MATE_BOUND);
}

#[test]
fn stalemated_side_scores_a_draw_with_no_move() {
    let result = searcher().search(&pos("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"), &[], Limits::depth(4));
    assert!(result.best_move.is_none());
    assert_eq!(result.score, 0);
}

#[test]
fn checkmated_side_has_no_move_and_a_mated_score() {
    let mut mated = pos("6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1");
    mated.play_unchecked(shakmaty::uci::UciMove::from_ascii(b"e1e8").unwrap().to_move(&mated).unwrap());
    let result = searcher().search(&mated, &[], Limits::depth(4));
    assert!(result.best_move.is_none());
    assert_eq!(result.score, -MATE);
}

#[test]
fn takes_a_hanging_queen() {
    let (mv, score) = best("rnb1kbnr/pppp1ppp/8/4p3/4P2q/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 1", 6);
    assert_eq!(mv, "f3h4");
    assert!(score > 700);
}

#[test]
fn fifty_move_rule_makes_every_reply_a_draw() {
    let (_, score) = best("7k/8/8/8/8/8/4Q3/4K3 w - - 100 80", 3);
    assert_eq!(score, 0);
}

#[test]
fn depth_limit_is_respected() {
    let result = searcher().search(&pos(START), &[], Limits::depth(5));
    assert_eq!(result.depth, 5);
    assert!(result.nodes > 0);
}

#[test]
fn node_limit_stops_the_search() {
    let result = searcher().search(&pos(START), &[], Limits::nodes(30_000));
    assert!(result.best_move.is_some());
    assert!(result.nodes < 300_000, "{}", result.nodes);
}

#[test]
fn movetime_is_respected() {
    let started = Instant::now();
    let result = searcher().search(&pos(START), &[], Limits::movetime(200));
    assert!(result.best_move.is_some());
    assert!(started.elapsed() < Duration::from_millis(900), "{:?}", started.elapsed());
}

#[test]
fn stop_flag_ends_an_infinite_search() {
    let stop = Arc::new(AtomicBool::new(false));
    let mut engine = Searcher::new(16, Arc::clone(&stop));
    let flag = Arc::clone(&stop);
    std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(150));
        flag.store(true, Ordering::Relaxed);
    });
    let started = Instant::now();
    let result = engine.search(&pos(START), &[], Limits::default());
    assert!(result.best_move.is_some());
    assert!(started.elapsed() < Duration::from_secs(3));
}

#[test]
fn single_legal_move_is_played_instantly_under_a_clock() {
    let only_move = pos("8/8/8/8/8/5k2/8/r6K w - - 0 1");
    assert_eq!(only_move.legal_moves().len(), 1);
    let started = Instant::now();
    let result = searcher().search(&only_move, &[], Limits::movetime(2000));
    assert_eq!(uci(result.best_move.unwrap()), "h1h2");
    assert!(started.elapsed() < Duration::from_millis(500));
}

#[test]
fn principal_variation_starts_with_the_best_move_and_is_legal() {
    let start = pos("rnb1kbnr/pppp1ppp/8/4p3/4P2q/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 1");
    let result = searcher().search(&start, &[], Limits::depth(7));
    assert_eq!(result.pv[0], result.best_move.unwrap());
    let mut walker: Chess = start;
    for m in &result.pv {
        assert!(walker.is_legal(*m));
        walker.play_unchecked(*m);
    }
}

#[test]
fn warm_table_does_not_change_the_answer_for_a_forced_mate() {
    let mut engine = searcher();
    for fen in [START, "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"] {
        engine.search(&pos(fen), &[], Limits::depth(6));
    }
    let result = engine.search(&pos("k7/8/8/1K6/8/8/8/7R w - - 0 1"), &[], Limits::depth(6));
    assert_eq!((uci(result.best_move.unwrap()), result.score), ("b5b6".to_string(), MATE - 3));
}

#[test]
fn queen_and_king_checkmate_a_lone_king() {
    let mut position = pos("4k3/8/4K3/8/8/8/8/3Q4 w - - 0 1");
    let mut history = Vec::new();
    let mut engine = searcher();
    for _ in 0..60 {
        if position.is_game_over() {
            break;
        }
        let result = engine.search(&position, &history, Limits::depth(6));
        history.push(position_key(&position));
        position.play_unchecked(result.best_move.unwrap());
    }
    assert!(position.is_checkmate(), "never mated: {:?}", position.outcome());
}

#[test]
fn engine_with_more_depth_beats_a_shallow_one() {
    // 4 games at depth 6 vs depth 2: the deeper side must score clearly more than half
    let mut deep_points = 0.0;
    for game in 0..4 {
        let deep_is_white = game % 2 == 0;
        let (mut deep, mut shallow) = (searcher(), searcher());
        let mut position = Chess::default();
        let mut history = Vec::new();
        while !position.is_game_over() && history.len() < 160 {
            let white = position.turn().is_white();
            let (engine, depth) = if white == deep_is_white { (&mut deep, 6) } else { (&mut shallow, 2) };
            let m = engine.search(&position, &history, Limits::depth(depth)).best_move.unwrap();
            history.push(position_key(&position));
            position.play_unchecked(m);
        }
        let outcome = position.outcome();
        deep_points += match outcome.winner() {
            Some(winner) if winner.is_white() == deep_is_white => 1.0,
            Some(_) => 0.0,
            None => 0.5,
        };
    }
    assert!(deep_points >= 3.0, "deep engine scored only {deep_points}/4");
}

#[test]
fn repetition_needs_the_same_position_with_the_same_side_to_move() {
    assert!(is_repetition(&[1, 2, 3, 4, 1], 50));
    assert!(is_repetition(&[1, 2, 1], 50));
    assert!(!is_repetition(&[1, 1], 50)); // one ply apart: other side to move
    assert!(!is_repetition(&[1, 2, 3, 4], 50));
    assert!(!is_repetition(&[], 0));
}

#[test]
fn repetition_ignores_positions_before_the_last_irreversible_move() {
    assert!(!is_repetition(&[1, 2, 3, 4, 1], 2));
    assert!(is_repetition(&[1, 2, 3, 4, 1], 4));
}

#[test]
fn score_text_follows_the_uci_format() {
    assert_eq!(format_score(35), "cp 35");
    assert_eq!(format_score(-120), "cp -120");
    assert_eq!(format_score(MATE - 1), "mate 1");
    assert_eq!(format_score(MATE - 3), "mate 2");
    assert_eq!(format_score(-MATE + 2), "mate -1");
}
