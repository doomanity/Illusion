mod common;

use std::time::Duration;

use common::{pos, START};
use illusion_engine::book::Book;
use illusion_engine::search::{should_stop_iterating, time_scale};
use shakmaty::{CastlingMode, Position};

#[test]
fn built_in_book_has_many_positions() {
    assert!(Book::built_in().positions() > 100);
}

#[test]
fn start_position_gets_a_first_move() {
    let book = Book::built_in();
    let start = pos(START);
    for random in 0..50u64 {
        let mv = book.probe(&start, random.wrapping_mul(0x9E37_79B9_7F4A_7C15)).unwrap();
        let uci = mv.to_uci(CastlingMode::Standard).to_string();
        assert!(["e2e4", "d2d4", "c2c4", "g1f3"].contains(&uci.as_str()), "{uci}");
    }
}

#[test]
fn every_book_move_is_legal() {
    let book = Book::built_in();
    let start = pos(START);
    let mv = book.probe(&start, 7).unwrap();
    assert!(start.legal_moves().contains(&mv));
}

#[test]
fn unknown_position_has_no_book_move() {
    let odd = pos("r3k2r/pppq1ppp/2n5/3pP3/8/2P2N2/PP3PPP/R2QK2R w KQkq - 0 12");
    assert!(Book::built_in().probe(&odd, 1).is_none());
}

#[test]
fn parse_ignores_move_numbers_and_comments() {
    let book = Book::parse("# a comment\n1. e4 e5 2. Nf3 Nc6\n\n").unwrap();
    assert_eq!(book.positions(), 4);
}

#[test]
fn parse_rejects_illegal_san() {
    assert!(Book::parse("1. e4 e5 2. Qh8").is_err());
}

#[test]
fn repeated_lines_make_a_move_more_likely() {
    let book = Book::parse("1. e4\n1. e4\n1. e4\n1. d4\n").unwrap();
    let start = pos(START);
    let mut e4 = 0;
    for random in 0..400u64 {
        let mv = book.probe(&start, random.wrapping_mul(0x9E37_79B9_7F4A_7C15)).unwrap();
        if mv.to_uci(CastlingMode::Standard).to_string() == "e2e4" {
            e4 += 1;
        }
    }
    assert!(e4 > 200 && e4 < 400, "e4 chosen {e4} of 400 times");
}

#[test]
fn shallow_iterations_use_the_full_budget() {
    assert_eq!(time_scale(5, 10, 0), 1.0);
}

#[test]
fn stable_best_move_shortens_the_search() {
    assert_eq!(time_scale(12, 4, 0), 0.5);
    assert_eq!(time_scale(12, 2, 0), 0.8);
    assert_eq!(time_scale(12, 0, 0), 1.0);
}

#[test]
fn score_drop_extends_the_search() {
    assert_eq!(time_scale(12, 0, 50), 1.5);
    assert_eq!(time_scale(12, 4, 50), 0.75);
}

#[test]
fn iteration_stops_when_next_one_would_not_fit() {
    let soft = Duration::from_millis(1000);
    assert!(!should_stop_iterating(Duration::from_millis(500), soft, None, 1.0));
    assert!(should_stop_iterating(Duration::from_millis(600), soft, None, 1.0));
    assert!(should_stop_iterating(Duration::from_millis(300), soft, None, 0.5));
}

#[test]
fn hard_limit_caps_an_extended_budget() {
    let soft = Duration::from_millis(1000);
    let hard = Some(Duration::from_millis(1200));
    assert!(should_stop_iterating(Duration::from_millis(720), soft, hard, 1.5));
}

#[test]
fn pass_through_moves_are_walked_but_never_played() {
    // a generated repertoire: we play 1.e4 and 2.Qh5, the human's 1...e5 is never a book move
    let book = Book::parse("e4 ~e5 Qh5\ne4 ~c5 Nf3\n").unwrap();
    assert_eq!(book.positions(), 3);
    let after_e4 = pos("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1");
    assert!(book.probe(&after_e4, 1).is_none());
    let after_e5 = pos("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2");
    let mv = book.probe(&after_e5, 1).unwrap();
    assert_eq!(mv.to_uci(CastlingMode::Standard).to_string(), "d1h5");
}

#[test]
fn load_reads_a_file_and_reports_missing_ones() {
    let path = std::env::temp_dir().join(format!("illusion_book_{}.txt", std::process::id()));
    std::fs::write(&path, "# test\ne4 ~e5 Nf3\n").unwrap();
    assert_eq!(Book::load(&path).unwrap().positions(), 2);
    std::fs::remove_file(&path).unwrap();
    assert!(Book::load(&path).is_err());
}
