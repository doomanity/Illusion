//! UCI protocol front end, plus time management and a `bench` command.

use std::io::{self, BufRead};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use shakmaty::fen::Fen;
use shakmaty::uci::UciMove;
use shakmaty::{CastlingMode, Chess, Color, EnPassantMode, Position};

use crate::book::Book;
use crate::brilliant::{self, DEFAULT_HUNT, MAX_HUNT};
use crate::eval;
use crate::nnue;
use crate::search::{format_score, position_key, Limits, Ponder, SearchResult, Searcher};
use crate::style::{Style, DEFAULT_AGGRESSION, DEFAULT_CONTEMPT, MAX_AGGRESSION, MAX_CONTEMPT};
use crate::swindle::{self, Settings, DEFAULT_OPPONENT_ELO, DEFAULT_TOLERANCE, SACRIFICE_HUNT_BONUS};

pub const NAME: &str = "Illusion";
pub const DEFAULT_HASH_MB: usize = 64;
pub const DEFAULT_OVERHEAD_MS: u64 = 30;
const SEARCH_STACK_BYTES: usize = 64 * 1024 * 1024;
const DEFAULT_MOVES_TO_GO: u64 = 20;
/// Extra time for traps and sacrifice checks after the full normal search, as a share of
/// the move's soft budget (they also use whatever the normal search left unused).
const EXTRA_SHARE_FOR_TRAPS: f64 = 0.3;

/// Cheap unpredictable number for picking among book moves (not for anything security related).
fn random_u64() -> u64 {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos() as u64);
    let mut z = nanos.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

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
    /// Searching on the opponent's time; the clock starts at `ponderhit`.
    pub ponder: bool,
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
            "infinite" => params.infinite = true,
            "ponder" => params.ponder = true,
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
    book: Book,
    use_book: bool,
    swindle: Option<Settings>,
    swindle_settings: Settings,
    ponder: Option<Arc<Ponder>>,
    human_mode: bool,
    style: Style,
    brilliant_hunt: i32,
    anglerfish: bool,
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
            book: Book::built_in(),
            use_book: true,
            swindle: None,
            swindle_settings: Settings::default(),
            ponder: None,
            human_mode: false,
            style: Style::default(),
            brilliant_hunt: DEFAULT_HUNT,
            anglerfish: true,
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
                println!("option name OwnBook type check default true");
                println!("option name BookFile type string default <empty>");
                println!("option name Ponder type check default false");
                println!("option name EvalFile type string default <empty>");
                println!("option name KingSafety type check default {}", eval::king_safety_enabled());
                println!("option name EndgameRules type check default {}", eval::endgame_rules_enabled());
                println!("option name HumanMode type check default false");
                println!("option name Aggression type spin default {DEFAULT_AGGRESSION} min 0 max {MAX_AGGRESSION}");
                println!("option name Contempt type spin default {DEFAULT_CONTEMPT} min 0 max {MAX_CONTEMPT}");
                println!("option name BrilliantHunt type spin default {DEFAULT_HUNT} min 0 max {MAX_HUNT}");
                println!("option name Anglerfish type check default true");
                println!("option name Swindle type check default false");
                println!("option name OpponentElo type spin default {DEFAULT_OPPONENT_ELO} min 400 max 2800");
                println!("option name SwindleTolerance type spin default {DEFAULT_TOLERANCE} min 0 max 300");
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
            "ponderhit" => {
                if let Some(ponder) = &self.ponder {
                    ponder.hit();
                }
            }
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
        let on = value.eq_ignore_ascii_case("true");
        match name.as_str() {
            "ownbook" => self.use_book = on,
            "bookfile" => self.load_book(&value),
            "humanmode" => {
                self.wait_for_search();
                self.human_mode = on;
                self.searcher.lock().unwrap().new_game(); // old scores were made with the other style
            }
            "aggression" => {
                if let Ok(n) = value.parse::<i32>() {
                    self.style.aggression = n.clamp(0, MAX_AGGRESSION);
                }
            }
            "anglerfish" => self.anglerfish = on,
            "brillianthunt" => {
                if let Ok(n) = value.parse::<i32>() {
                    self.brilliant_hunt = n.clamp(0, MAX_HUNT);
                }
            }
            "contempt" => {
                if let Ok(n) = value.parse::<i32>() {
                    self.style.contempt = n.clamp(0, MAX_CONTEMPT);
                }
            }
            "kingsafety" => eval::set_king_safety(on),
            "evalfile" => {
                self.wait_for_search();
                load_network(&value);
            }
            "endgamerules" => eval::set_endgame_rules(on),
            "swindle" => self.swindle = on.then_some(self.swindle_settings),
            "opponentelo" | "swindletolerance" => {
                let Ok(number) = value.parse::<i32>() else { return };
                if name == "opponentelo" {
                    self.swindle_settings.opponent_elo = number.clamp(400, 2800);
                } else {
                    self.swindle_settings.tolerance = number.clamp(0, 300);
                }
                if self.swindle.is_some() {
                    self.swindle = Some(self.swindle_settings);
                }
            }
            _ => {}
        }
        match (name.as_str(), value.parse::<usize>()) {
            ("hash", Ok(mb)) => {
                self.wait_for_search();
                self.searcher.lock().unwrap().tt.resize(mb.clamp(1, 4096));
            }
            ("move overhead", Ok(ms)) => self.overhead_ms = ms as u64,
            _ => {}
        }
    }

    /// Swap in a book file (empty = back to the built-in book). A broken file keeps the old book.
    fn load_book(&mut self, path: &str) {
        let path = path.trim();
        if path.is_empty() || path == "<empty>" {
            self.book = Book::built_in();
            return;
        }
        match Book::load(std::path::Path::new(path)) {
            Ok(book) => {
                println!("info string loaded book {path} ({} positions)", book.positions());
                self.book = book;
            }
            Err(message) => println!("info string {message}"),
        }
    }

    /// A book move, if the book is on, this is a normal timed game search and the position is known.
    fn book_move(&self, params: &GoParams) -> Option<String> {
        let analysing = params.infinite || params.ponder || params.depth.is_some() || params.nodes.is_some();
        if !self.use_book || analysing {
            return None;
        }
        let mv = self.book.probe(&self.pos, random_u64())?;
        Some(mv.to_uci(CastlingMode::Standard).to_string())
    }

    fn go(&mut self, args: &[&str]) {
        self.wait_for_search();
        let params = parse_go(args);
        if let Some(book_move) = self.book_move(&params) {
            println!("info string book move");
            println!("bestmove {book_move}");
            return;
        }
        let limits = limits_for(&params, self.pos.turn(), self.overhead_ms);
        // trap mode only in real games on our own clock: it splits the clock, and analysis wants the true best move
        // human mode switches trap mode on too, with sacrifice hunting
        let anglerfish = self.anglerfish;
        let traps = match (self.swindle, self.human_mode) {
            (_, true) => Some(Settings { sacrifice_bonus: SACRIFICE_HUNT_BONUS, anglerfish, ..self.swindle_settings }),
            (swindle, false) => swindle.map(|s| Settings { anglerfish, ..s }),
        };
        let on_clock = limits.soft_time.is_some() && !params.infinite && !params.ponder;
        let swindle = traps.filter(|_| on_clock);
        // like traps, sacrifices get less room against stronger opponents (only mates at 2600+)
        let room = Settings { tolerance: self.brilliant_hunt, ..self.swindle_settings };
        let hunt = (self.human_mode && on_clock && self.brilliant_hunt > 0).then(|| swindle::effective_tolerance(room));
        let style = self.human_mode.then_some((self.style, self.pos.turn()));
        let ponder = params.ponder.then(|| Arc::new(Ponder::new()));
        self.ponder = ponder.clone();
        let (pos, history) = (self.pos.clone(), self.history.clone());
        let (searcher, stop) = (Arc::clone(&self.searcher), Arc::clone(&self.stop));
        self.stop.store(false, Ordering::Relaxed);
        let worker = std::thread::Builder::new().stack_size(SEARCH_STACK_BYTES).spawn(move || {
            let mut searcher = searcher.lock().unwrap();
            searcher.ponder = ponder.clone();
            searcher.style = style;
            let (best, reply) = if swindle.is_some() || hunt.is_some() {
                search_with_traps(&mut searcher, &pos, &history, limits, swindle, hunt)
            } else {
                let result = searcher.search(&pos, &history, limits);
                (result.best_move, result.pv.get(1).copied())
            };
            searcher.ponder = None;
            searcher.style = None;
            // a pondering engine must not answer before the opponent has actually moved
            if let Some(ponder) = &ponder {
                while ponder.is_active() && !stop.load(Ordering::Relaxed) {
                    std::thread::sleep(Duration::from_millis(1));
                }
            }
            let uci = |m: shakmaty::Move| m.to_uci(CastlingMode::Standard).to_string();
            match (best, reply) {
                (Some(m), Some(r)) => println!("bestmove {} ponder {}", uci(m), uci(r)),
                (Some(m), None) => println!("bestmove {}", uci(m)),
                (None, _) => println!("bestmove 0000"),
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

/// Load the network at `path`; an empty path or `<empty>` goes back to the hand-written eval.
fn load_network(path: &str) {
    let path = path.trim();
    if path.is_empty() || path == "<empty>" {
        nnue::set_active(None);
        println!("info string using hand-written eval");
        return;
    }
    match nnue::Network::load(std::path::Path::new(path)) {
        Ok(network) => {
            nnue::set_active(Some(network));
            println!("info string loaded network {path}");
        }
        Err(message) => {
            nnue::set_active(None);
            println!("info string {message}, using hand-written eval");
        }
    }
}

/// Normal search on part of the time, then spend the rest looking for a sound sacrifice
/// (`hunt` = how many centipawns it may cost, `None` = off) and a trap for the opponent.
fn search_with_traps(
    searcher: &mut Searcher,
    pos: &Chess,
    history: &[u64],
    limits: Limits,
    traps: Option<Settings>,
    hunt: Option<i32>,
) -> (Option<shakmaty::Move>, Option<shakmaty::Move>) {
    let traps = traps.filter(|&settings| swindle::effective_tolerance(settings) > 0);
    if traps.is_none() && hunt.is_none() {
        let result = searcher.search(pos, history, limits);
        return (result.best_move, result.pv.get(1).copied());
    }
    let start = Instant::now();
    let soft = limits.soft_time.expect("trap mode needs a clock");
    // the normal search keeps its whole budget: cutting it costs a ply and far more elo than traps win
    let main: SearchResult = searcher.search(pos, history, limits.clone());
    let latest = start + limits.hard_time.unwrap_or(soft);
    let deadline = (start + soft).max(Instant::now() + soft.mul_f64(EXTRA_SHARE_FOR_TRAPS)).min(latest);
    let main_reply = main.pv.get(1).copied();
    let name = |m: shakmaty::Move| m.to_uci(CastlingMode::Standard).to_string();

    if main.best_move.is_some_and(|m| brilliant::is_sacrifice(pos, m)) {
        println!("info string brilliant move !! (the best move is a sacrifice)");
        return (main.best_move, main_reply);
    }

    let printing = std::mem::replace(&mut searcher.print_info, false);
    let sacrifice = hunt.and_then(|max_loss| brilliant::hunt(searcher, pos, history, &main, max_loss, deadline));
    let choice = match (sacrifice, traps) {
        (None, Some(settings)) => swindle::choose(searcher, pos, history, &main, settings, deadline),
        _ => None,
    };
    searcher.print_info = printing;

    if let Some((mv, score)) = sacrifice {
        println!(
            "info string brilliant move !! sacrifice {} ({}) over {} ({})",
            name(mv),
            format_score(score),
            main.best_move.map_or_else(String::new, name),
            format_score(main.score)
        );
        return (Some(mv), None);
    }
    let Some(choice) = choice else { return (main.best_move, main_reply) };
    if Some(choice.mv) != main.best_move {
        println!(
            "info string trap {} (expected {:+.0}) over {} (expected {:+.0}) vs elo {}",
            name(choice.mv),
            choice.expected,
            main.best_move.map_or_else(String::new, name),
            choice.best_expected,
            traps.map_or(0, |s| s.opponent_elo)
        );
    }
    if let (true, Some(reply)) = (Some(choice.mv) != main.best_move, choice.bait) {
        println!("info string anglerfish bait {}: {} looks good for them but loses", name(choice.mv), name(reply));
    }
    if choice.brilliant {
        println!("info string brilliant move !!");
    }
    let reply = if Some(choice.mv) == main.best_move { main_reply } else { None };
    (Some(choice.mv), reply)
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
