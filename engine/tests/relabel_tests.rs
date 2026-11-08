mod common;

use std::path::PathBuf;

use common::pos;
use illusion_engine::datagen::{Outcome, Sample, SAMPLE_BYTES};
use illusion_engine::relabel::{parse_score, relabel, sample_fen, RelabelConfig, MAX_TEACHER_SCORE};
use shakmaty::Position;

const WHITE_QUEEN_UP_BLACK_TO_MOVE: &str = "4k3/8/8/8/8/8/8/3QK3 b - - 0 1";
const BLACK_QUEEN_UP_WHITE_TO_MOVE: &str = "3qk3/8/8/8/8/8/8/4K3 w - - 0 1";

fn sample(fen: &str, score: i32, outcome: Outcome) -> Sample {
    let position = pos(fen);
    Sample::new(position.board(), position.turn(), score, outcome)
}

fn temp(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("illusion-relabel-{}-{name}", std::process::id()))
}

fn teacher() -> RelabelConfig {
    RelabelConfig { teacher: PathBuf::from(env!("CARGO_BIN_EXE_illusion")), nodes: 2000, threads: 2 }
}

#[test]
fn scores_are_read_from_info_lines() {
    assert_eq!(parse_score("info depth 12 seldepth 18 score cp -37 nodes 100 pv e2e4"), Some(-37));
    assert_eq!(parse_score("info depth 20 score mate 3 pv a1a8"), Some(MAX_TEACHER_SCORE));
    assert_eq!(parse_score("info depth 20 score mate -2"), Some(-MAX_TEACHER_SCORE));
    assert_eq!(parse_score("info depth 9 score cp 99999 lowerbound"), Some(MAX_TEACHER_SCORE));
    assert_eq!(parse_score("info string hello"), None);
    assert_eq!(parse_score("bestmove e2e4"), None);
}

#[test]
fn sample_fen_rebuilds_the_position() {
    let original = pos("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b - - 0 1");
    let rebuilt = pos(&sample_fen(&Sample::new(original.board(), original.turn(), 0, Outcome::Draw)));
    assert_eq!(rebuilt.board(), original.board());
    assert_eq!(rebuilt.turn(), original.turn());
}

#[test]
fn relabel_keeps_positions_and_results_but_uses_teacher_scores() {
    let (input, output) = (temp("in1.bin"), temp("out1.bin"));
    let originals = [
        sample(WHITE_QUEEN_UP_BLACK_TO_MOVE, 0, Outcome::WhiteWins),
        sample(BLACK_QUEEN_UP_WHITE_TO_MOVE, 0, Outcome::BlackWins),
        sample(common::START, 500, Outcome::Draw),
    ];
    let bytes: Vec<u8> = originals.iter().flat_map(|s| s.to_bytes()).collect();
    std::fs::write(&input, &bytes).unwrap();
    std::fs::remove_file(&output).ok();

    relabel(&input, &output, &teacher()).unwrap();
    let out = std::fs::read(&output).unwrap();
    std::fs::remove_file(&input).ok();
    std::fs::remove_file(&output).ok();

    let relabeled: Vec<Sample> = out.chunks_exact(SAMPLE_BYTES).map(|r| Sample::from_bytes(r).unwrap()).collect();
    assert_eq!(relabeled.len(), originals.len());
    for (new, old) in relabeled.iter().zip(&originals) {
        assert_eq!((new.occupied, new.pieces, new.outcome, new.side_to_move), (old.occupied, old.pieces, old.outcome, old.side_to_move));
    }
    assert!(relabeled[0].score > 500, "white is a queen up: {}", relabeled[0].score);
    assert!(relabeled[1].score < -500, "black is a queen up: {}", relabeled[1].score);
    assert!(i32::from(relabeled[2].score).abs() < 100, "start position: {}", relabeled[2].score);
}

#[test]
fn a_stopped_run_resumes_where_it_left_off() {
    let (input, output) = (temp("in2.bin"), temp("out2.bin"));
    let samples: Vec<Sample> = (0..5000).map(|i| sample(common::START, i % 7, Outcome::Draw)).collect();
    let bytes: Vec<u8> = samples.iter().flat_map(|s| s.to_bytes()).collect();
    std::fs::write(&input, &bytes).unwrap();
    // pretend 3000 were done (marked with a score nobody would produce) plus half a record
    let mut done: Vec<u8> = samples[..3000].iter().flat_map(|s| {
        let mut marked = s.clone();
        marked.score = 1234;
        marked.to_bytes()
    }).collect();
    done.extend_from_slice(&[0u8; 10]);
    std::fs::write(&output, &done).unwrap();

    let config = RelabelConfig { nodes: 50, ..teacher() };
    relabel(&input, &output, &config).unwrap();
    let out = std::fs::read(&output).unwrap();
    std::fs::remove_file(&input).ok();
    std::fs::remove_file(&output).ok();

    assert_eq!(out.len(), bytes.len());
    let scores: Vec<i16> = out.chunks_exact(SAMPLE_BYTES).map(|r| Sample::from_bytes(r).unwrap().score).collect();
    assert!(scores[..3000].iter().all(|&s| s == 1234));
    assert!(scores[3000..].iter().all(|&s| s != 1234));
}

#[test]
fn a_missing_teacher_is_a_clear_error() {
    let (input, output) = (temp("in3.bin"), temp("out3.bin"));
    std::fs::write(&input, sample(common::START, 0, Outcome::Draw).to_bytes()).unwrap();
    let config = RelabelConfig { teacher: PathBuf::from("/no/such/engine"), ..teacher() };
    let error = relabel(&input, &output, &config).unwrap_err();
    std::fs::remove_file(&input).ok();
    std::fs::remove_file(&output).ok();
    assert!(error.to_string().contains("cannot start"), "{error}");
}
