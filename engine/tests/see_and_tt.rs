mod common;

use common::pos;
use illusion_engine::see::see;
use illusion_engine::tt::{Bound, Tt};
use shakmaty::uci::UciMove;
use shakmaty::Position;

fn see_of(fen: &str, uci: &str) -> i32 {
    let position = pos(fen);
    let m = UciMove::from_ascii(uci.as_bytes()).unwrap().to_move(&position).unwrap();
    see(position.board(), m)
}

#[test]
fn queen_taking_a_defended_pawn_loses_the_queen() {
    assert_eq!(see_of("4k3/8/3p4/4p3/8/8/4Q3/4K3 w - - 0 1", "e2e5"), 100 - 900);
}

#[test]
fn queen_taking_a_loose_pawn_wins_a_pawn() {
    assert_eq!(see_of("4k3/8/8/4p3/8/8/4Q3/4K3 w - - 0 1", "e2e5"), 100);
}

#[test]
fn even_knight_trade_is_zero() {
    assert_eq!(see_of("4k3/8/5n2/3n4/8/2N5/8/4K3 w - - 0 1", "c3d5"), 0);
}

#[test]
fn pawn_takes_defended_knight_nets_a_knight_for_a_pawn() {
    assert_eq!(see_of("4k3/8/3p4/4n3/3P4/8/8/4K3 w - - 0 1", "d4e5"), 320 - 100);
}

#[test]
fn quiet_moves_and_castling_score_zero() {
    assert_eq!(see_of(common::START, "e2e4"), 0);
    assert_eq!(see_of("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1g1"), 0);
}

#[test]
fn en_passant_wins_a_pawn() {
    assert_eq!(see_of("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", "e5d6"), 100);
}

#[test]
fn x_ray_attackers_join_the_exchange() {
    // two rooks stacked on the e-file against a pawn defended once: Rxe5 Rxe5 Rxe5 wins a pawn
    assert_eq!(see_of("4r1k1/8/8/4p3/8/8/4R3/4R1K1 w - - 0 1", "e2e5"), 100);
}

#[test]
fn table_stores_and_returns_entries() {
    let mut tt = Tt::new(1);
    assert!(tt.probe(42).is_none());
    tt.store(42, None, 150, 7, Bound::Exact);
    let entry = tt.probe(42).unwrap();
    assert_eq!((entry.score, entry.depth, entry.bound), (150, 7, Bound::Exact));
}

#[test]
fn table_ignores_other_positions_in_the_same_slot() {
    let mut tt = Tt::new(1);
    tt.store(5, None, 10, 3, Bound::Lower);
    let colliding = 5 + (1u64 << 40); // same low bits, different key
    assert!(tt.probe(colliding).is_none());
}

#[test]
fn table_keeps_a_much_deeper_result_for_the_same_position() {
    let mut tt = Tt::new(1);
    tt.store(9, None, 100, 12, Bound::Exact);
    tt.store(9, None, -50, 3, Bound::Exact);
    assert_eq!(tt.probe(9).unwrap().score, 100);
}

#[test]
fn clear_empties_the_table() {
    let mut tt = Tt::new(1);
    tt.store(1, None, 1, 1, Bound::Exact);
    tt.clear();
    assert!(tt.probe(1).is_none());
}
