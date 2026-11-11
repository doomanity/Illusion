mod common;

use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::{Duration, Instant};

use common::pos;
use illusion_engine::brilliant::{hunt, is_sacrifice};
use illusion_engine::search::{Limits, Searcher};
use shakmaty::uci::UciMove;
use shakmaty::{CastlingMode, Position};

fn mv(fen: &str, uci: &str) -> (shakmaty::Chess, shakmaty::Move) {
    let position = pos(fen);
    let m = uci.parse::<UciMove>().unwrap().to_move(&position).unwrap();
    (position, m)
}

#[test]
fn giving_a_piece_for_a_pawn_is_a_sacrifice() {
    // Bxh7+ Kxh7: bishop for a pawn
    let (position, m) = mv("r1bq1rk1/pppn1ppp/4pn2/3p4/1bPP4/2NBPN2/PP3PPP/R1BQK2R w KQ - 0 7", "d3h7");
    assert!(is_sacrifice(&position, m));
}

#[test]
fn ordinary_moves_and_fair_trades_are_not_sacrifices() {
    let (position, quiet) = mv(common::START, "e2e4");
    assert!(!is_sacrifice(&position, quiet));
    let (position, trade) = mv("4k3/8/3p4/4n3/8/5N2/8/4K3 w - - 0 1", "f3e5");
    assert!(!is_sacrifice(&position, trade));
}

#[test]
fn hunting_off_finds_nothing() {
    let position = pos("r4rk1/1bqnbppp/p2ppn2/1p6/3NP3/1BN1BP2/PPPQ2PP/2KR3R w - - 0 12");
    let mut searcher = Searcher::new(16, Arc::new(AtomicBool::new(false)));
    let main = searcher.search(&position, &[], Limits::depth(8));
    let deadline = Instant::now() + Duration::from_secs(5);
    assert!(hunt(&mut searcher, &position, &[], &main, 0, deadline).is_none());
}

#[test]
fn any_sacrifice_found_holds_up_under_a_deeper_look() {
    let max_loss = 30;
    for fen in [
        "r4rk1/1bqnbppp/p2ppn2/1p6/3NP3/1BN1BP2/PPPQ2PP/2KR3R w - - 0 12",
        "r1bq1rk1/pppn1ppp/4pn2/3p4/1bPP4/2NBPN2/PP3PPP/R1BQK2R w KQ - 0 7",
        "2r2rk1/pp1bqpp1/2n1pn1p/3p4/3P4/2PBPN2/P1QN1PPP/R4RK1 w - - 0 14",
    ] {
        let position = pos(fen);
        let mut searcher = Searcher::new(16, Arc::new(AtomicBool::new(false)));
        let main = searcher.search(&position, &[], Limits::depth(9));
        let deadline = Instant::now() + Duration::from_secs(20);
        let Some((m, _)) = hunt(&mut searcher, &position, &[], &main, max_loss, deadline) else { continue };
        assert!(is_sacrifice(&position, m));
        let after = position.clone().play(m).unwrap();
        let deeper = -searcher.search(&after, &[], Limits::depth(10)).score;
        let slack = 60; // different depths see slightly different things
        assert!(
            deeper >= main.score - max_loss - slack,
            "{fen}: {} scores {deeper} vs best {}",
            m.to_uci(CastlingMode::Standard),
            main.score
        );
    }
}

#[test]
fn hunting_respects_its_deadline() {
    let position = pos("r4rk1/1bqnbppp/p2ppn2/1p6/3NP3/1BN1BP2/PPPQ2PP/2KR3R w - - 0 12");
    let mut searcher = Searcher::new(16, Arc::new(AtomicBool::new(false)));
    let main = searcher.search(&position, &[], Limits::depth(8));
    let started = Instant::now();
    let _ = hunt(&mut searcher, &position, &[], &main, 100, started + Duration::from_millis(100));
    assert!(started.elapsed() < Duration::from_millis(400), "{:?}", started.elapsed());
}
