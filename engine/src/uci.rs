//! UCI protocol front end, plus time management and a `bench` command.

use std::io::{self, BufRead};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use shakmaty::fen::Fen;
use shakmaty::uci::UciMove;
use shakmaty::{CastlingMode, Chess, Color, EnPassantMode, Position};

use crate::search::{position_key, Limits, Searcher};

pub const NAME: &str = "Illusion";
pub const DEFAULT_HASH_MB: usize = 64;
pub const DEFAULT_OVERHEAD_MS: u64 = 30;
const SEARCH_STACK_BYTES: usize = 64 * 1024 * 1024;
const DEFAULT_MOVES_TO_GO: u64 = 30;

/// How long to spend on one move: (soft limit, hard limit).
///
/// The soft limit stops us starting a new iteration, the hard limit aborts mid-search.
pub fn allocate_time(time_ms: u64, inc_ms: u64, moves_to_go: Option<u32>, overhead_ms: u64) -> (Duration, Duration) {
    let time = time_ms.saturating_sub(overhead_ms).max(1);
    let moves_left = moves_to_go.map_or(DEFAULT_MOVES_TO_GO, |m| u64::from(m.clamp(1, 50)));
    let base = time / moves_left + inc_ms * 3 / 4;
    let soft = base.min(time / 2).max(1);
    let hard = (soft * 3).min(time * 7 / 10).max(soft);
    (Duration::from_millis(soft), Duration::from_millis(hard))
}

#[derive(Debug, Default, PartialEq)]
pub struct GoParams {
    pub wtime: Option<u64>,
    pub btime: Option<u64>,
    pub winc: u64,
    pub binc: u64,
    pub moves_to_go: Option<u32>,
    pub movetime: Option<u64>,
    pub depth: Option<i32>,
    pub nodes: Option<u64>,
    pub infinite: bool,
}

pub fn parse_go(tokens: &[&str]) -> GoParams {
    let mut params = GoParams::default();
    let mut iter = tokens.iter();
    while let Some(&token) = iter.next() {
        let mut number = || iter.next().and_then(|value| value.parse::<u64>().ok());
        match token {
            "wtime" => params.wtime = number(),
            "btime" => params.btime = number(),
            "winc" => params.winc = number().unwrap_or(0),
            "binc" => params.binc = number().unwrap_or(0),
            "movestogo" => params.moves_to_go = number().map(|v| v as u32),
            "movetime" => params.movetime = number(),
            "depth" => params.depth = number().map(|v| v as i32),
            "nodes" => params.nodes = number(),
            "infinite" | "ponder" => params.infinite = true,
            _ => {}
        }
    }
    params
}

pub fn limits_for(params: &GoParams, side: Color, overhead_ms: u64) -> Limits {
    let mut limits = Limits { max_depth: params.depth, max_nodes: params.nodes, ..Limits::default() };
    if let Some(ms) = params.movetime {
        let ms = ms.saturating_sub(overhead_ms).max(1);
        limits.soft_time = Some(Duration::from_millis(ms));
        limits.hard_time = Some(Duration::from_millis(ms));
    } else if !params.infinite {
        let (time, inc) = if side.is_white() { (params.wtime, params.winc) } else { (params.btime, params.binc) };
        if let Some(time) = time {
            let (soft, hard) = allocate_time(time, inc, params.moves_to_go, overhead_ms);
            limits.soft_time = Some(soft);
            limits.hard_time = Some(hard);
        }
    }
    limits
}

/// Parse the arguments of a `position` command. Returns the position and the keys of
/// every earlier position in the game (needed for repetition detection).
pub fn parse_position(tokens: &[&str]) -> Result<(Chess, Vec<u64>), String> {
    let moves_at = tokens.iter().position(|&t| t == "moves").unwrap_or(tokens.len());
    let mut pos = match tokens.first() {
        Some(&"startpos") => Chess::default(),
        Some(&"fen") => {
            let fen = tokens[1..moves_at].join(" ");
            Fen::from_ascii(fen.as_bytes())
                .map_err(|e| format!("bad fen: {e}"))?
                .into_position(CastlingMode::Standard)
                .map_err(|e| format!("illegal position: {e}"))?
        }
        _ => return Err("expected startpos or fen".to_string()),
    };
    let mut history = Vec::new();
    for text in tokens.iter().skip(moves_at + 1) {
        let mv = UciMove::from_ascii(text.as_bytes())
            .map_err(|e| format!("bad move {text}: {e}"))?
            .to_move(&pos)
            .map_err(|e| format!("illegal move {text}: {e}"))?;
        history.push(position_key(&pos));
        pos.play_unchecked(mv);
    }
    Ok((pos, history))
}

pub struct Uci {
    pos: Chess,
    history: Vec<u64>,
    searcher: Arc<Mutex<Searcher>>,
    stop: Arc<AtomicBool>,
    worker: Option<JoinHandle<()>>,
    overhead_ms: u64,
}

impl Default for Uci {
    fn default() -> Self {
        Uci::new()
    }
}

impl Uci {
    pub fn new() -> Uci {
        let stop = Arc::new(AtomicBool::new(false));
        let mut searcher = Searcher::new(DEFAULT_HASH_MB, Arc::clone(&stop));
        searcher.print_info = true;
        Uci {
            pos: Chess::default(),
            history: Vec::new(),
            searcher: Arc::new(Mutex::new(searcher)),
            stop,
            worker: None,
            overhead_ms: DEFAULT_OVERHEAD_MS,
        }
    }

    /// Handle one command line. Returns false when the engine should quit.
    pub fn handle(&mut self, line: &str) -> bool {
        let tokens: Vec<&str> = line.split_whitespace().collect();
        let Some((&command, args)) = tokens.split_first() else { return true };
        match command {
            "uci" => {
                println!("id name {NAME}");
                println!("id author Brody and Claude");
                println!("option name Hash type spin default {DEFAULT_HASH_MB} min 1 max 4096");
                println!("option name Move Overhead type spin default {DEFAULT_OVERHEAD_MS} min 0 max 5000");
                println!("uciok");
            }
            "isready" => println!("readyok"),
            "ucinewgame" => {
                self.wait_for_search();
                self.searcher.lock().unwrap().new_game();
                self.pos = Chess::default();
                self.history.clear();
            }
            "position" => {
                self.wait_for_search();
                match parse_position(args) {
                    Ok((pos, history)) => {
                        self.pos = pos;
                        self.history = history;
                    }
                    Err(message) => println!("info string {message}"),
                }
            }
            "setoption" => self.set_option(args),
            "go" => self.go(args),
            "stop" => self.stop.store(true, Ordering::Relaxed),
            "d" => println!("{}", Fen::from_position(&self.pos, EnPassantMode::Legal)),
            "quit" => {
                self.stop.store(true, Ordering::Relaxed);
                self.wait_for_search();
                return false;
            }
            _ => {}
        }
        true
    }

    fn set_option(&mut self, args: &[&str]) {
        let value_at = args.iter().position(|&t| t == "value");
        let Some(value_at) = value_at else { return };
        let name = args[1..value_at].join(" ").to_lowercase();
        let value = args[value_at + 1..].join(" ");
        match (name.as_str(), value.parse::<usize>()) {
            ("hash", Ok(mb)) => {
                self.wait_for_search();
                self.searcher.lock().unwrap().tt.resize(mb.clamp(1, 4096));
            }
            ("move overhead", Ok(ms)) => self.overhead_ms = ms as u64,
            _ => {}
        }
    }

    fn go(&mut self, args: &[&str]) {
        self.wait_for_search();
        let limits = limits_for(&parse_go(args), self.pos.turn(), self.overhead_ms);
        let (pos, history) = (self.pos.clone(), self.history.clone());
        let searcher = Arc::clone(&self.searcher);
        self.stop.store(false, Ordering::Relaxed);
        let worker = std::thread::Builder::new().stack_size(SEARCH_STACK_BYTES).spawn(move || {
            let result = searcher.lock().unwrap().search(&pos, &history, limits);
            match result.best_move {
                Some(m) => println!("bestmove {}", m.to_uci(CastlingMode::Standard)),
                None => println!("bestmove 0000"),
            }
        });
        self.worker = worker.ok();
    }

    fn wait_for_search(&mut self) {
        if let Some(handle) = self.worker.take() {
            let _ = handle.join();
        }
    }
}

pub fn run() {
    let mut uci = Uci::new();
    for line in io::stdin().lock().lines() {
        let Ok(line) = line else { break };
        if !uci.handle(&line) {
            break;
        }
    }
    uci.stop.store(true, Ordering::Relaxed);
    uci.wait_for_search();
}

const BENCH_FENS: [&str; 6] = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "r2q1rk1/ppp2ppp/2n1bn2/2bpp3/4P3/2NP1N2/PPP1BPPP/R1BQ1RK1 w - - 0 8",
    "2rr3k/pp3pp1/1nnqbN1p/3pN3/2pP4/2P3Q1/PPB4P/R4RK1 w - - 0 1",
    "6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1",
];

/// Search a fixed set of positions and report speed. Returns (nodes, milliseconds).
pub fn bench(depth: i32) -> (u64, u128) {
    let stop = Arc::new(AtomicBool::new(false));
    let mut searcher = Searcher::new(DEFAULT_HASH_MB, stop);
    let started = Instant::now();
    let mut nodes = 0;
    for fen in BENCH_FENS {
        searcher.new_game();
        let pos: Chess = Fen::from_ascii(fen.as_bytes()).unwrap().into_position(CastlingMode::Standard).unwrap();
        nodes += searcher.search(&pos, &[], Limits::depth(depth)).nodes;
    }
    (nodes, started.elapsed().as_millis())
}
