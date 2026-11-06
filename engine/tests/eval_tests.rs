mod common;

use common::{pos, START};
use illusion_engine::eval::evaluate;

/// Same position with colours swapped and the board flipped vertically.
fn mirror(fen: &str) -> String {
    let parts: Vec<&str> = fen.split(' ').collect();
    let swap_case = |c: char| if c.is_ascii_uppercase() { c.to_ascii_lowercase() } else { c.to_ascii_uppercase() };
    let board: Vec<String> = parts[0].split('/').rev().map(|rank| rank.chars().map(swap_case).collect()).collect();
    let turn = if parts[1] == "w" { "b" } else { "w" };
    let castling: String = if parts[2] == "-" { "-".into() } else { parts[2].chars().map(swap_case).collect() };
    let ep = if parts[3] == "-" {
        "-".to_string()
    } else {
        let (file, rank) = parts[3].split_at(1);
        format!("{file}{}", if rank == "3" { "6" } else { "3" })
    };
    format!("{} {turn} {castling} {ep} {} {}", board.join("/"), parts[4], parts[5])
}

const POSITIONS: [&str; 7] = [
    START,
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "r2q1rk1/ppp2ppp/2n1bn2/2bpp3/4P3/2NP1N2/PPP1BPPP/R1BQ1RK1 w - - 0 8",
    "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1",
    "rnbqkb1r/pp1ppppp/5n2/2p5/4P3/2N5/PPPP1PPP/R1BQKBNR w KQkq - 2 3",
    "6k1/5ppp/8/8/8/8/5PPP/3R2K1 b - - 0 1",
];

#[test]
fn evaluation_is_colour_symmetric() {
    for fen in POSITIONS {
        assert_eq!(evaluate(&pos(fen)), evaluate(&pos(&mirror(fen))), "{fen}");
    }
}

#[test]
fn starting_position_is_roughly_equal_with_a_tempo_bonus() {
    let score = evaluate(&pos(START));
    assert!((0..40).contains(&score), "{score}");
}

#[test]
fn extra_queen_is_worth_more_than_an_extra_pawn() {
    let queen_up = evaluate(&pos("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"));
    let pawn_up = evaluate(&pos("rnbqkbnr/ppppppp1/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"));
    assert!(queen_up > pawn_up + 600);
}

#[test]
fn side_to_move_flips_the_sign() {
    let white = evaluate(&pos("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"));
    let black = evaluate(&pos("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1"));
    assert!(white > 800 && black < -700);
}

#[test]
fn passed_pawns_are_valued_more_the_further_they_advance() {
    let far = evaluate(&pos("4k3/8/4P3/8/8/8/8/4K3 w - - 0 1"));
    let near = evaluate(&pos("4k3/8/8/8/8/4P3/8/4K3 w - - 0 1"));
    assert!(far > near);
}
