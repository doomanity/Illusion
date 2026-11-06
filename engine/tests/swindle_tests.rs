mod common;

use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::{Duration, Instant};

use common::pos;
use illusion_engine::search::{Limits, Searcher};
use illusion_engine::swindle::{choose, reply_probabilities, temperature, Settings};
use shakmaty::Position;

const MIDDLEGAMES: [&str; 3] = [
    "r1b2rk1/pp1nqppp/2pbpn2/3p4/2PP4/2NBPN2/PPQ2PPP/R1B2RK1 w - - 4 9",
    "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 1 5",
    "r2q1rk1/ppp2ppp/2np1n2/2b1p1B1/2B1P1b1/2NP1N2/PPP2PPP/R2Q1RK1 w - - 4 8",
];

fn searcher() -> Searcher {
    Searcher::new(16, Arc::new(AtomicBool::new(false)))
}

#[test]
fn weaker_players_are_noisier() {
    assert!(temperature(800) > temperature(1500));
    assert!(temperature(1500) > temperature(2200));
    assert_eq!(temperature(100), temperature(600));
    assert_eq!(temperature(3000), temperature(2400));
}

#[test]
fn probabilities_add_up_to_one() {
    let p = reply_probabilities(&[50.0, -100.0, -400.0], &[0.0, 1.0, 0.0], 1200);
    assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-9);
}

#[test]
fn strong_players_find_the_best_reply_more_often() {
    let quality = [0.0, -150.0, -150.0, -150.0];
    let instinct = [0.0; 4];
    let weak = reply_probabilities(&quality, &instinct, 800)[0];
    let strong = reply_probabilities(&quality, &instinct, 2200)[0];
    assert!(strong > weak + 0.3, "strong {strong} weak {weak}");
}

#[test]
fn tempting_moves_attract_weak_players() {
    let p = reply_probabilities(&[0.0, 0.0], &[0.0, 1.0], 800);
    assert!(p[1] > p[0]);
}

#[test]
fn already_winning_means_no_tricks() {
    let queen_up = pos("4k3/8/8/8/8/8/3QK3/8 w - - 0 1");
    let mut s = searcher();
    let main = s.search(&queen_up, &[], Limits::depth(6));
    let deadline = Instant::now() + Duration::from_secs(5);
    assert!(choose(&mut s, &queen_up, &[], &main, Settings::default(), deadline).is_none());
}

#[test]
fn trap_moves_stay_close_to_the_best_move() {
    let settings = Settings { opponent_elo: 1000, tolerance: 60 };
    for fen in MIDDLEGAMES {
        let position = pos(fen);
        let mut s = searcher();
        let main = s.search(&position, &[], Limits::depth(9));
        let deadline = Instant::now() + Duration::from_secs(20);
        let Some(choice) = choose(&mut s, &position, &[], &main, settings, deadline) else { continue };
        assert!(position.legal_moves().contains(&choice.mv));
        let after = position.clone().play(choice.mv).unwrap();
        let score = -s.search(&after, &[], Limits::depth(9)).score;
        let slack = 40; // the deep check and the trap search see different depths
        assert!(score >= main.score - settings.tolerance - slack, "{fen}: {score} vs best {}", main.score);
    }
}

#[test]
fn trap_search_respects_its_deadline() {
    let position = pos(MIDDLEGAMES[0]);
    let mut s = searcher();
    let main = s.search(&position, &[], Limits::depth(8));
    let started = Instant::now();
    let _ = choose(&mut s, &position, &[], &main, Settings::default(), started + Duration::from_millis(100));
    assert!(started.elapsed() < Duration::from_millis(400));
}
