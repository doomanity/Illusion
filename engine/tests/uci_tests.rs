mod common;

use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{self, Receiver};
use std::time::{Duration, Instant};

use common::{is_legal_uci, pos, START};
use illusion_engine::uci::{allocate_time, limits_for, parse_go, parse_position, GoParams};
use shakmaty::{Color, Position};

struct Engine {
    child: Child,
    stdin: ChildStdin,
    lines: Receiver<String>,
}

impl Engine {
    fn start() -> Engine {
        let mut child = Command::new(env!("CARGO_BIN_EXE_illusion"))
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .expect("engine binary should start");
        let stdin = child.stdin.take().unwrap();
        let stdout = child.stdout.take().unwrap();
        let (sender, lines) = mpsc::channel();
        std::thread::spawn(move || {
            for line in BufReader::new(stdout).lines().map_while(Result::ok) {
                if sender.send(line).is_err() {
                    break;
                }
            }
        });
        Engine { child, stdin, lines }
    }

    fn send(&mut self, line: &str) {
        writeln!(self.stdin, "{line}").unwrap();
        self.stdin.flush().unwrap();
    }

    /// True if no line starting with `prefix` arrives within `wait`.
    fn stays_quiet(&mut self, prefix: &str, wait: Duration) -> bool {
        let deadline = Instant::now() + wait;
        loop {
            let left = deadline.saturating_duration_since(Instant::now());
            match self.lines.recv_timeout(left) {
                Ok(line) if line.starts_with(prefix) => return false,
                Ok(_) => {}
                Err(_) => return true,
            }
        }
    }

    /// Read lines until one starts with `prefix`; fail after `timeout`.
    fn wait_for(&mut self, prefix: &str, timeout: Duration) -> String {
        let deadline = Instant::now() + timeout;
        loop {
            let left = deadline.saturating_duration_since(Instant::now());
            match self.lines.recv_timeout(left) {
                Ok(line) if line.starts_with(prefix) => return line,
                Ok(_) => {}
                Err(_) => panic!("timed out waiting for '{prefix}'"),
            }
        }
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        let _ = writeln!(self.stdin, "quit");
        let _ = self.child.wait();
    }
}

const WAIT: Duration = Duration::from_secs(10);

#[test]
fn answers_the_uci_handshake() {
    let mut engine = Engine::start();
    engine.send("uci");
    assert_eq!(engine.wait_for("id name", WAIT), "id name Illusion");
    engine.wait_for("uciok", WAIT);
    engine.send("isready");
    engine.wait_for("readyok", WAIT);
}

#[test]
fn plays_a_legal_move_at_fixed_depth() {
    let mut engine = Engine::start();
    engine.send("position startpos moves e2e4 e7e5");
    engine.send("go depth 5");
    let reply = engine.wait_for("bestmove", WAIT);
    let text = reply.split_whitespace().nth(1).unwrap();
    let (after, _) = parse_position(&["startpos", "moves", "e2e4", "e7e5"]).unwrap();
    assert!(is_legal_uci(&after, text), "{reply}");
}

#[test]
fn prints_search_info_with_a_score_and_pv() {
    let mut engine = Engine::start();
    engine.send("position startpos");
    engine.send("go depth 6");
    let info = engine.wait_for("info depth 6", WAIT);
    assert!(info.contains(" score cp ") && info.contains(" nodes ") && info.contains(" pv "), "{info}");
}

#[test]
fn respects_movetime() {
    let mut engine = Engine::start();
    engine.send("position startpos");
    let started = Instant::now();
    engine.send("go movetime 300");
    engine.wait_for("bestmove", WAIT);
    assert!(started.elapsed() < Duration::from_millis(1200), "{:?}", started.elapsed());
}

#[test]
fn uses_the_clock_in_a_blitz_game() {
    let mut engine = Engine::start();
    engine.send("position startpos");
    let started = Instant::now();
    engine.send("go wtime 3000 btime 3000 winc 0 binc 0");
    engine.wait_for("bestmove", WAIT);
    assert!(started.elapsed() < Duration::from_millis(1000), "{:?}", started.elapsed());
}

#[test]
fn stop_ends_an_infinite_search_with_a_move() {
    let mut engine = Engine::start();
    engine.send("position startpos");
    engine.send("go infinite");
    std::thread::sleep(Duration::from_millis(300));
    engine.send("stop");
    let reply = engine.wait_for("bestmove", Duration::from_secs(3));
    assert!(reply.split_whitespace().nth(1).is_some_and(|m| m != "0000"), "{reply}");
}

#[test]
fn finds_a_forced_mate_through_the_protocol() {
    let mut engine = Engine::start();
    engine.send("position fen k7/8/8/1K6/8/8/8/7R w - - 0 1");
    engine.send("go depth 6");
    let line = engine.wait_for("bestmove", WAIT);
    assert_eq!(line.split_whitespace().nth(1), Some("b5b6"), "{line}");
}

#[test]
fn works_across_several_games_and_option_changes() {
    let mut engine = Engine::start();
    engine.send("setoption name Hash value 16");
    engine.send("setoption name Move Overhead value 10");
    for _ in 0..2 {
        engine.send("ucinewgame");
        engine.send("position startpos moves d2d4");
        engine.send("go depth 4");
        engine.wait_for("bestmove", WAIT);
    }
    engine.send("isready");
    engine.wait_for("readyok", WAIT);
}

#[test]
fn survives_garbage_input() {
    let mut engine = Engine::start();
    engine.send("position fen not a fen");
    engine.send("position startpos moves e2e5");
    engine.send("blah blah");
    engine.send("go");
    engine.send("isready");
    engine.wait_for("readyok", WAIT);
}

#[test]
fn parses_castling_promotion_and_en_passant() {
    let (castled, _) = parse_position(&["startpos", "moves", "e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6", "e1g1"]).unwrap();
    assert!(castled.board().piece_at(shakmaty::Square::G1).is_some_and(|p| p.role == shakmaty::Role::King));

    let (promoted, _) = parse_position(&["fen", "8/P6k/8/8/8/8/8/K7", "w", "-", "-", "0", "1", "moves", "a7a8q"]).unwrap();
    assert!(promoted.board().piece_at(shakmaty::Square::A8).is_some_and(|p| p.role == shakmaty::Role::Queen));

    let (ep, history) =
        parse_position(&["startpos", "moves", "e2e4", "a7a6", "e4e5", "d7d5", "e5d6"]).unwrap();
    assert!(ep.board().piece_at(shakmaty::Square::D5).is_none());
    assert_eq!(history.len(), 5);
}

#[test]
fn rejects_illegal_moves_and_bad_fens() {
    assert!(parse_position(&["startpos", "moves", "e2e5"]).is_err());
    assert!(parse_position(&["fen", "nonsense"]).is_err());
    assert!(parse_position(&[]).is_err());
}

#[test]
fn parses_go_parameters() {
    let params = parse_go(&["wtime", "60000", "btime", "59000", "winc", "1000", "binc", "500", "movestogo", "20", "depth", "12"]);
    assert_eq!(
        params,
        GoParams {
            wtime: Some(60000),
            btime: Some(59000),
            winc: 1000,
            binc: 500,
            moves_to_go: Some(20),
            depth: Some(12),
            ..GoParams::default()
        }
    );
    assert!(parse_go(&["infinite"]).infinite);
    let pondering = parse_go(&["ponder", "wtime", "1000", "btime", "1000"]);
    assert!(pondering.ponder && !pondering.infinite);
    assert_eq!(parse_go(&["movetime", "250"]).movetime, Some(250));
}

#[test]
fn limits_pick_the_right_side_clock() {
    let params = parse_go(&["wtime", "60000", "btime", "1000"]);
    let white = limits_for(&params, Color::White, 30).soft_time.unwrap();
    let black = limits_for(&params, Color::Black, 30).soft_time.unwrap();
    assert!(white > black * 10);
    assert!(limits_for(&parse_go(&["infinite"]), Color::White, 30).hard_time.is_none());
}

#[test]
fn time_allocation_is_sane_across_time_controls() {
    for (time, inc) in [(180_000, 0), (60_000, 0), (60_000, 1_000), (10_000, 0), (1_000, 0), (200, 0), (50, 0), (5, 0)] {
        let (soft, hard) = allocate_time(time, inc, None, 30);
        assert!(soft >= Duration::from_millis(1), "{time}");
        assert!(hard >= soft, "{time}");
        assert!(hard <= Duration::from_millis(time.max(1)), "never plans to use more than the clock: {time}");
    }
    let (soft, _) = allocate_time(60_000, 0, None, 30);
    assert!(soft >= Duration::from_millis(2500) && soft <= Duration::from_millis(3500));
    let (with_inc, _) = allocate_time(60_000, 2_000, None, 30);
    assert!(with_inc > soft);
}

#[test]
fn startpos_fen_roundtrip_matches() {
    let from_startpos = parse_position(&["startpos"]).unwrap().0;
    assert_eq!(from_startpos, pos(START));
}

#[test]
fn start_position_with_a_clock_answers_from_the_book() {
    let mut engine = Engine::start();
    engine.send("position startpos");
    engine.send("go wtime 60000 btime 60000");
    let started = Instant::now();
    let line = engine.wait_for("bestmove", Duration::from_secs(5));
    assert!(started.elapsed() < Duration::from_millis(500));
    let mv = line.split_whitespace().nth(1).unwrap();
    assert!(["e2e4", "d2d4", "c2c4", "g1f3"].contains(&mv), "{mv}");
}

#[test]
fn fixed_depth_search_ignores_the_book() {
    let mut engine = Engine::start();
    engine.send("position startpos");
    engine.send("go depth 3");
    let line = engine.wait_for("bestmove", Duration::from_secs(10));
    assert!(line.starts_with("bestmove "));
}

#[test]
fn own_book_can_be_switched_off() {
    let mut engine = Engine::start();
    engine.send("setoption name OwnBook value false");
    engine.send("position startpos");
    engine.send("go movetime 200");
    engine.wait_for("info depth", Duration::from_secs(5));
    engine.wait_for("bestmove", Duration::from_secs(5));
}

#[test]
fn trap_options_are_advertised() {
    let mut engine = Engine::start();
    engine.send("uci");
    engine.wait_for("option name Swindle type check default false", Duration::from_secs(5));
    engine.wait_for("option name OpponentElo", Duration::from_secs(5));
    engine.wait_for("option name SwindleTolerance", Duration::from_secs(5));
    engine.wait_for("uciok", Duration::from_secs(5));
}

#[test]
fn trap_mode_plays_a_legal_move_on_time() {
    let fen = "r1b2rk1/pp1nqppp/2pbpn2/3p4/2PP4/2NBPN2/PPQ2PPP/R1B2RK1 w - - 4 9";
    let mut engine = Engine::start();
    engine.send("setoption name Swindle value true");
    engine.send("setoption name OpponentElo value 1000");
    engine.send(&format!("position fen {fen}"));
    let started = Instant::now();
    engine.send("go movetime 500");
    let line = engine.wait_for("bestmove", Duration::from_secs(5));
    assert!(started.elapsed() < Duration::from_millis(900));
    let mv = line.split_whitespace().nth(1).unwrap();
    assert!(is_legal_uci(&pos(fen), mv), "{mv}");
}

#[test]
fn eval_feature_toggles_are_advertised() {
    let mut engine = Engine::start();
    engine.send("uci");
    engine.wait_for("option name KingSafety type check default false", Duration::from_secs(5));
    engine.wait_for("option name EndgameRules type check default true", Duration::from_secs(5));
    engine.send("setoption name KingSafety value true");
    engine.send("position startpos");
    engine.send("go depth 4");
    engine.wait_for("bestmove", Duration::from_secs(5));
}

#[test]
fn best_move_comes_with_an_expected_reply() {
    let mut engine = Engine::start();
    engine.send("uci");
    engine.wait_for("option name Ponder type check", Duration::from_secs(5));
    engine.send("position startpos moves e2e4");
    engine.send("go depth 8");
    let line = engine.wait_for("bestmove", Duration::from_secs(10));
    let words: Vec<&str> = line.split_whitespace().collect();
    assert_eq!(words.len(), 4, "{line}");
    assert_eq!(words[2], "ponder");
    let position = illusion_engine::uci::parse_position(&["startpos", "moves", "e2e4", words[1]]).unwrap().0;
    assert!(is_legal_uci(&position, words[3]), "e2e4 {} then {}", words[1], words[3]);
}

#[test]
fn pondering_waits_for_the_opponent_then_uses_its_own_clock() {
    let mut engine = Engine::start();
    engine.send("position startpos moves e2e4 e7e5");
    engine.send("go ponder wtime 10000 btime 10000");
    assert!(engine.stays_quiet("bestmove", Duration::from_millis(1500)), "answered while pondering");
    engine.send("ponderhit");
    let started = Instant::now();
    engine.wait_for("bestmove", Duration::from_secs(5));
    // our budget for a 10 s clock is well under 2 s once the clock starts
    assert!(started.elapsed() < Duration::from_secs(2), "took {:?} after ponderhit", started.elapsed());
}

#[test]
fn stop_ends_pondering_with_a_move() {
    let mut engine = Engine::start();
    engine.send("position startpos moves d2d4");
    engine.send("go ponder wtime 60000 btime 60000");
    assert!(engine.stays_quiet("bestmove", Duration::from_millis(300)));
    engine.send("stop");
    engine.wait_for("bestmove", Duration::from_secs(2));
}

#[test]
fn a_finished_ponder_search_still_waits_for_the_opponent() {
    // only one legal move: the search ends at once, but the answer must wait for ponderhit
    let mut engine = Engine::start();
    engine.send("position fen 7k/8/8/8/8/8/1r6/K1r5 w - - 0 1");
    engine.send("go ponder wtime 60000 btime 60000");
    assert!(engine.stays_quiet("bestmove", Duration::from_millis(500)));
    engine.send("ponderhit");
    engine.wait_for("bestmove", Duration::from_secs(2));
}

#[test]
fn human_mode_options_work_and_still_play_legal_moves_on_time() {
    let fen = "r1b2rk1/pp1nqppp/2pbpn2/3p4/2PP4/2NBPN2/PPQ2PPP/R1B2RK1 w - - 4 9";
    let mut engine = Engine::start();
    engine.send("uci");
    engine.wait_for("option name HumanMode type check default false", Duration::from_secs(5));
    engine.wait_for("option name Aggression", Duration::from_secs(5));
    engine.wait_for("option name Contempt", Duration::from_secs(5));
    engine.send("setoption name HumanMode value true");
    engine.send("setoption name Aggression value 100");
    engine.send("setoption name OpponentElo value 1200");
    engine.send(&format!("position fen {fen}"));
    let started = Instant::now();
    engine.send("go wtime 10000 btime 10000 winc 100 binc 100");
    let line = engine.wait_for("bestmove", Duration::from_secs(5));
    assert!(started.elapsed() < Duration::from_secs(3));
    let mv = line.split_whitespace().nth(1).unwrap();
    assert!(is_legal_uci(&pos(fen), mv), "{mv}");
}
