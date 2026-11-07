//! Iterative deepening principal variation search.
//!
//! Techniques: transposition table, aspiration windows, check extensions, null move pruning,
//! reverse futility, futility and late move pruning, late move reductions, killer and
//! history move ordering, SEE-ordered captures and a quiescence search.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use shakmaty::zobrist::Zobrist64;
use shakmaty::{CastlingMode, Chess, Color, EnPassantMode, Move, Position, Role};

use crate::eval::{adjust_network_score, evaluate, is_pawn_ending, MATE, MATE_BOUND};
use crate::nnue::{self, Accumulator, Network};
use crate::see::{role_value, see};
use crate::tt::{Bound, Tt};

pub const MAX_PLY: usize = 128;
pub const INF: i32 = 32_000;

const ASPIRATION_START_DEPTH: i32 = 5;
const ASPIRATION_WINDOW: i32 = 25;
const NODES_PER_CLOCK_CHECK: u64 = 2048;
const HISTORY_LIMIT: i32 = 16_384;

const TT_MOVE_SCORE: i32 = 1_000_000;
const GOOD_CAPTURE_SCORE: i32 = 200_000;
const PROMOTION_SCORE: i32 = 150_000;
const KILLER_SCORES: [i32; 2] = [90_000, 80_000];
const BAD_CAPTURE_SCORE: i32 = -50_000;

#[derive(Clone, Debug, Default)]
pub struct Limits {
    pub max_depth: Option<i32>,
    pub max_nodes: Option<u64>,
    /// Stop starting new iterations after about this long.
    pub soft_time: Option<Duration>,
    /// Abort the search immediately after this long.
    pub hard_time: Option<Duration>,
}

impl Limits {
    pub fn depth(depth: i32) -> Limits {
        Limits { max_depth: Some(depth), ..Limits::default() }
    }

    pub fn nodes(nodes: u64) -> Limits {
        Limits { max_nodes: Some(nodes), ..Limits::default() }
    }

    pub fn movetime(ms: u64) -> Limits {
        let time = Some(Duration::from_millis(ms));
        Limits { soft_time: time, hard_time: time, ..Limits::default() }
    }

    fn is_timed(&self) -> bool {
        self.soft_time.is_some() || self.hard_time.is_some()
    }
}

#[derive(Clone, Debug)]
pub struct SearchResult {
    pub best_move: Option<Move>,
    pub score: i32,
    pub depth: i32,
    pub nodes: u64,
    pub pv: Vec<Move>,
}

pub fn position_key(pos: &Chess) -> u64 {
    pos.zobrist_hash::<Zobrist64>(EnPassantMode::PseudoLegal).0
}

/// Has the position at the end of `path` already occurred since the last irreversible move?
pub fn is_repetition(path: &[u64], halfmoves: u32) -> bool {
    let Some((&current, earlier)) = path.split_last() else { return false };
    // Only positions with the same side to move (every second entry) can repeat.
    earlier
        .iter()
        .rev()
        .take(halfmoves as usize)
        .skip(1)
        .step_by(2)
        .any(|&key| key == current)
}

fn to_tt_score(score: i32, ply: usize) -> i32 {
    if score >= MATE_BOUND {
        score + ply as i32
    } else if score <= -MATE_BOUND {
        score - ply as i32
    } else {
        score
    }
}

fn from_tt_score(score: i32, ply: usize) -> i32 {
    if score >= MATE_BOUND {
        score - ply as i32
    } else if score <= -MATE_BOUND {
        score + ply as i32
    } else {
        score
    }
}

fn mvv_lva(m: Move) -> i32 {
    let victim = if m.is_en_passant() { role_value(Role::Pawn) } else { m.capture().map_or(0, role_value) };
    victim * 10 - m.role() as i32
}

fn is_quiet(m: Move) -> bool {
    !m.is_capture() && !m.is_promotion()
}

fn lmr_table() -> [[i32; 64]; 64] {
    let mut table = [[0; 64]; 64];
    for (depth, row) in table.iter_mut().enumerate().skip(1) {
        for (count, cell) in row.iter_mut().enumerate().skip(1) {
            *cell = (0.75 + (depth as f64).ln() * (count as f64).ln() / 2.25) as i32;
        }
    }
    table
}

fn color_index(color: Color) -> usize {
    if color.is_white() {
        0
    } else {
        1
    }
}

const STABLE_ITERATIONS_FOR_SHORT_TIME: u32 = 4;
const STABLE_ITERATIONS_FOR_MEDIUM_TIME: u32 = 2;
const MIN_DEPTH_FOR_EARLY_STOP: i32 = 8;
const SCORE_DROP_THAT_EXTENDS_TIME: i32 = 30;

/// How much of the soft time budget to use: less when the best move keeps coming out the
/// same, more when the score just dropped and the engine may be in trouble.
pub fn time_scale(depth: i32, stable_iterations: u32, score_drop: i32) -> f64 {
    let mut scale = if depth < MIN_DEPTH_FOR_EARLY_STOP {
        1.0
    } else if stable_iterations >= STABLE_ITERATIONS_FOR_SHORT_TIME {
        0.5
    } else if stable_iterations >= STABLE_ITERATIONS_FOR_MEDIUM_TIME {
        0.8
    } else {
        1.0
    };
    if score_drop >= SCORE_DROP_THAT_EXTENDS_TIME {
        scale *= 1.5;
    }
    scale
}

/// Share of the budget after which a new iteration is not started (it would likely overrun).
const START_ITERATION_SHARE: f64 = 0.6;

/// Is it too late to start another iteration?
pub fn should_stop_iterating(elapsed: Duration, soft: Duration, hard: Option<Duration>, scale: f64) -> bool {
    let mut budget = soft.mul_f64(scale);
    if let Some(hard) = hard {
        budget = budget.min(hard);
    }
    elapsed >= budget.mul_f64(START_ITERATION_SHARE)
}

/// UCI score text: `cp 35` or `mate 3` (negative when we are being mated).
pub fn format_score(score: i32) -> String {
    if score >= MATE_BOUND {
        format!("mate {}", (MATE - score + 1) / 2)
    } else if score <= -MATE_BOUND {
        format!("mate -{}", (MATE + score) / 2)
    } else {
        format!("cp {score}")
    }
}

pub struct Searcher {
    pub tt: Tt,
    pub print_info: bool,
    stop: Arc<AtomicBool>,
    nodes: u64,
    start: Instant,
    limits: Limits,
    can_abort: bool,
    aborted: bool,
    seldepth: usize,
    killers: [[Option<Move>; 2]; MAX_PLY],
    history: Box<[[[i32; 64]; 64]; 2]>,
    path: Vec<u64>,
    root_best: Option<Move>,
    lmr: [[i32; 64]; 64],
    network: Option<Arc<Network>>,
    accumulators: Vec<Accumulator>, // network inputs for the position at each ply
}

impl Searcher {
    pub fn new(hash_mb: usize, stop: Arc<AtomicBool>) -> Searcher {
        Searcher {
            tt: Tt::new(hash_mb),
            print_info: false,
            stop,
            nodes: 0,
            start: Instant::now(),
            limits: Limits::default(),
            can_abort: false,
            aborted: false,
            seldepth: 0,
            killers: [[None; 2]; MAX_PLY],
            history: Box::new([[[0; 64]; 64]; 2]),
            path: Vec::with_capacity(MAX_PLY + 512),
            root_best: None,
            lmr: lmr_table(),
            network: None,
            accumulators: vec![Accumulator::default(); MAX_PLY + 1],
        }
    }

    fn static_eval(&self, pos: &Chess, ply: usize) -> i32 {
        match &self.network {
            Some(network) => adjust_network_score(network.output(&self.accumulators[ply], pos.turn()), pos),
            None => evaluate(pos),
        }
    }

    /// Bring the network inputs for `child` (one ply below `parent`) up to date.
    fn prepare_child(&mut self, ply: usize, parent: &Chess, child: &Chess) {
        if let Some(network) = &self.network {
            let (done, next) = self.accumulators.split_at_mut(ply + 1);
            network.update(&done[ply], parent.board(), child.board(), &mut next[0]);
        }
    }

    /// Has the GUI asked us to stop?
    pub fn stopped(&self) -> bool {
        self.stop.load(Ordering::Relaxed)
    }

    pub fn new_game(&mut self) {
        self.tt.clear();
        self.history.iter_mut().flatten().for_each(|row| row.fill(0));
        self.killers = [[None; 2]; MAX_PLY];
    }

    /// Search `pos`. `game_history` holds the keys of every earlier position in the game.
    pub fn search(&mut self, pos: &Chess, game_history: &[u64], limits: Limits) -> SearchResult {
        self.start = Instant::now();
        self.network = nnue::active();
        if let Some(network) = &self.network {
            self.accumulators[0] = network.refresh(pos.board());
        }
        self.nodes = 0;
        self.aborted = false;
        self.can_abort = false;
        self.seldepth = 0;
        self.limits = limits;
        self.killers = [[None; 2]; MAX_PLY];
        self.path.clear();
        self.path.extend_from_slice(game_history);
        let root_key = position_key(pos);
        self.path.push(root_key);

        let legal = pos.legal_moves();
        let Some(&first) = legal.first() else {
            let score = if pos.is_check() { -MATE } else { 0 };
            return SearchResult { best_move: None, score, depth: 0, nodes: 0, pv: vec![] };
        };
        if legal.len() == 1 && self.limits.is_timed() {
            return SearchResult { best_move: Some(first), score: self.static_eval(pos, 0), depth: 0, nodes: 0, pv: vec![first] };
        }

        let max_depth = self.limits.max_depth.unwrap_or(MAX_PLY as i32 - 8).min(MAX_PLY as i32 - 8);
        let mut best_move = first;
        let mut best_score = -INF;
        let mut completed_depth = 0;
        let mut stable_iterations = 0u32;
        let mut previous_score: Option<i32> = None;
        self.root_best = Some(first);

        for depth in 1..=max_depth {
            self.can_abort = depth > 1; // always finish depth 1 so we have a move
            let score = self.aspirate(pos, root_key, depth, best_score);
            if self.aborted {
                break;
            }
            let score_drop = previous_score.map_or(0, |previous| previous - score);
            best_score = score;
            previous_score = Some(score);
            completed_depth = depth;
            if let Some(m) = self.root_best {
                stable_iterations = if m == best_move { stable_iterations + 1 } else { 0 };
                best_move = m;
            }
            if self.print_info {
                self.print_iteration(pos, depth, score);
            }
            if let Some(soft) = self.limits.soft_time {
                let scale = time_scale(depth, stable_iterations, score_drop);
                if should_stop_iterating(self.start.elapsed(), soft, self.limits.hard_time, scale) {
                    break; // the next iteration would not finish in time, or the move is settled
                }
            }
            if score.abs() >= MATE_BOUND && depth >= 4 && self.limits.is_timed() {
                break; // forced mate found, no need to keep searching
            }
        }

        SearchResult {
            best_move: Some(best_move),
            score: best_score,
            depth: completed_depth,
            nodes: self.nodes,
            pv: self.extract_pv(pos, best_move),
        }
    }

    fn aspirate(&mut self, pos: &Chess, key: u64, depth: i32, previous: i32) -> i32 {
        if depth < ASPIRATION_START_DEPTH {
            return self.negamax(pos, key, depth, 0, -INF, INF, true, false);
        }
        let mut delta = ASPIRATION_WINDOW;
        let mut alpha = (previous - delta).max(-INF);
        let mut beta = (previous + delta).min(INF);
        loop {
            let score = self.negamax(pos, key, depth, 0, alpha, beta, true, false);
            if self.aborted {
                return score;
            }
            if score <= alpha {
                alpha = (alpha - delta).max(-INF);
            } else if score >= beta {
                beta = (beta + delta).min(INF);
            } else {
                return score;
            }
            delta *= 2;
            if delta > 1000 {
                alpha = -INF;
                beta = INF;
            }
        }
    }

    fn should_abort(&mut self) -> bool {
        if self.aborted {
            return true;
        }
        if !self.can_abort || !self.nodes.is_multiple_of(NODES_PER_CLOCK_CHECK) {
            return false;
        }
        let out_of_time = self.limits.hard_time.is_some_and(|limit| self.start.elapsed() >= limit);
        let out_of_nodes = self.limits.max_nodes.is_some_and(|limit| self.nodes >= limit);
        if out_of_time || out_of_nodes || self.stop.load(Ordering::Relaxed) {
            self.aborted = true;
        }
        self.aborted
    }

    fn is_draw(&self, pos: &Chess) -> bool {
        pos.halfmoves() >= 100 || pos.is_insufficient_material() || is_repetition(&self.path, pos.halfmoves())
    }

    #[allow(clippy::too_many_arguments)]
    fn negamax(
        &mut self,
        pos: &Chess,
        key: u64,
        mut depth: i32,
        ply: usize,
        mut alpha: i32,
        mut beta: i32,
        pv_node: bool,
        allow_null: bool,
    ) -> i32 {
        self.nodes += 1;
        if self.should_abort() {
            return 0;
        }
        self.seldepth = self.seldepth.max(ply);
        if ply >= MAX_PLY - 1 {
            return self.static_eval(pos, ply);
        }
        if ply > 0 {
            if self.is_draw(pos) {
                return 0;
            }
            alpha = alpha.max(-MATE + ply as i32);
            beta = beta.min(MATE - ply as i32 - 1);
            if alpha >= beta {
                return alpha;
            }
        }

        let in_check = pos.is_check();
        if in_check {
            depth += 1;
        }
        if depth <= 0 {
            return self.quiesce(pos, ply, alpha, beta);
        }

        let mut tt_move = None;
        if let Some(entry) = self.tt.probe(key) {
            tt_move = entry.mv;
            if !pv_node && entry.depth >= depth {
                let score = from_tt_score(entry.score, ply);
                match entry.bound {
                    Bound::Exact => return score,
                    Bound::Lower if score >= beta => return score,
                    Bound::Upper if score <= alpha => return score,
                    _ => {}
                }
            }
        }

        let static_eval = if in_check { -INF } else { self.static_eval(pos, ply) };

        if !pv_node && !in_check {
            if depth <= 6 && beta.abs() < MATE_BOUND && static_eval - 90 * depth >= beta {
                return static_eval; // reverse futility: so far ahead that a search will not help
            }
            if allow_null && depth >= 3 && static_eval >= beta && !is_pawn_ending(pos, pos.turn()) {
                let reduction = 3 + depth / 4 + ((static_eval - beta) / 200).min(2);
                if let Ok(child) = pos.clone().swap_turn() {
                    self.prepare_child(ply, pos, &child);
                    let child_key = position_key(&child);
                    self.path.push(child_key);
                    let score =
                        -self.negamax(&child, child_key, depth - 1 - reduction, ply + 1, -beta, -beta + 1, false, false);
                    self.path.pop();
                    if self.aborted {
                        return 0;
                    }
                    if score >= beta {
                        return if score >= MATE_BOUND { beta } else { score };
                    }
                }
            }
        }

        let mut moves = pos.legal_moves();
        if moves.is_empty() {
            return if in_check { -MATE + ply as i32 } else { 0 };
        }
        let mut scores = [0i32; 256];
        for (i, &m) in moves.iter().enumerate() {
            scores[i] = self.score_move(pos, m, tt_move, ply);
        }

        let side = pos.turn();
        let original_alpha = alpha;
        let mut best = -INF;
        let mut best_move = None;
        let mut tried_quiets: Vec<Move> = Vec::new();
        let mut move_count = 0usize;

        for i in 0..moves.len() {
            let pick = (i..moves.len()).max_by_key(|&j| scores[j]).unwrap_or(i);
            moves.swap(i, pick);
            scores.swap(i, pick);
            let m = moves[i];
            let quiet = is_quiet(m);
            let bad_capture = scores[i] < 0 && m.is_capture();

            let mut child = pos.clone();
            child.play_unchecked(m);
            let gives_check = child.is_check();

            if !pv_node && !in_check && !gives_check && best > -MATE_BOUND && move_count > 0 {
                if quiet && depth <= 4 && move_count as i32 >= 3 + depth * depth {
                    continue; // late move pruning
                }
                if quiet && depth <= 3 && static_eval + 120 * depth + 60 <= alpha {
                    continue; // futility pruning
                }
                if bad_capture && depth <= 4 {
                    continue;
                }
            }

            self.prepare_child(ply, pos, &child);
            let child_key = position_key(&child);
            self.path.push(child_key);
            move_count += 1;
            let new_depth = depth - 1;

            let score = if move_count == 1 {
                -self.negamax(&child, child_key, new_depth, ply + 1, -beta, -alpha, pv_node, true)
            } else {
                let mut reduction = 0;
                if depth >= 3 && move_count > 2 && quiet && !in_check {
                    reduction = self.lmr[(depth as usize).min(63)][move_count.min(63)];
                    if pv_node {
                        reduction -= 1;
                    }
                    if gives_check {
                        reduction -= 1;
                    }
                    if self.killers[ply].contains(&Some(m)) {
                        reduction -= 1;
                    }
                    reduction = reduction.clamp(0, (new_depth - 1).max(0));
                }
                let mut score =
                    -self.negamax(&child, child_key, new_depth - reduction, ply + 1, -alpha - 1, -alpha, false, true);
                if score > alpha && reduction > 0 {
                    score = -self.negamax(&child, child_key, new_depth, ply + 1, -alpha - 1, -alpha, false, true);
                }
                if score > alpha && score < beta {
                    score = -self.negamax(&child, child_key, new_depth, ply + 1, -beta, -alpha, true, true);
                }
                score
            };
            self.path.pop();
            if self.aborted {
                return 0;
            }

            if score > best {
                best = score;
                best_move = Some(m);
                if score > alpha {
                    alpha = score;
                    if ply == 0 {
                        self.root_best = Some(m);
                    }
                    if alpha >= beta {
                        if quiet {
                            self.reward_quiet(side, m, &tried_quiets, depth, ply);
                        }
                        break;
                    }
                }
            }
            if quiet {
                tried_quiets.push(m);
            }
        }

        let bound = if best <= original_alpha {
            Bound::Upper
        } else if best >= beta {
            Bound::Lower
        } else {
            Bound::Exact
        };
        self.tt.store(key, best_move, to_tt_score(best, ply), depth, bound);
        best
    }

    fn quiesce(&mut self, pos: &Chess, ply: usize, mut alpha: i32, beta: i32) -> i32 {
        self.nodes += 1;
        if self.should_abort() {
            return 0;
        }
        self.seldepth = self.seldepth.max(ply);
        if ply >= MAX_PLY - 1 {
            return self.static_eval(pos, ply);
        }
        let in_check = pos.is_check();
        let mut best = if in_check { -INF } else { self.static_eval(pos, ply) };
        if !in_check {
            if best >= beta {
                return best;
            }
            alpha = alpha.max(best);
        }

        let mut moves = if in_check { pos.legal_moves() } else { pos.capture_moves() };
        if in_check && moves.is_empty() {
            return -MATE + ply as i32;
        }
        let mut scores = [0i32; 256];
        for (i, &m) in moves.iter().enumerate() {
            scores[i] = mvv_lva(m);
        }

        for i in 0..moves.len() {
            let pick = (i..moves.len()).max_by_key(|&j| scores[j]).unwrap_or(i);
            moves.swap(i, pick);
            scores.swap(i, pick);
            let m = moves[i];
            if !in_check {
                let victim = m.capture().map_or(0, role_value);
                if best + victim + 200 < alpha && !m.is_promotion() {
                    continue; // delta pruning: even winning the piece will not reach alpha
                }
                if see(pos.board(), m) < 0 {
                    continue;
                }
            }
            let mut child = pos.clone();
            child.play_unchecked(m);
            self.prepare_child(ply, pos, &child);
            let score = -self.quiesce(&child, ply + 1, -beta, -alpha);
            if self.aborted {
                return 0;
            }
            if score > best {
                best = score;
                if score > alpha {
                    alpha = score;
                    if alpha >= beta {
                        break;
                    }
                }
            }
        }
        best
    }

    fn score_move(&self, pos: &Chess, m: Move, tt_move: Option<Move>, ply: usize) -> i32 {
        if Some(m) == tt_move {
            return TT_MOVE_SCORE;
        }
        if m.is_capture() {
            let order = mvv_lva(m);
            return if see(pos.board(), m) >= 0 { GOOD_CAPTURE_SCORE + order } else { BAD_CAPTURE_SCORE + order };
        }
        if m.is_promotion() {
            return PROMOTION_SCORE;
        }
        if self.killers[ply][0] == Some(m) {
            return KILLER_SCORES[0];
        }
        if self.killers[ply][1] == Some(m) {
            return KILLER_SCORES[1];
        }
        match m.from() {
            Some(from) => self.history[color_index(pos.turn())][from as usize][m.to() as usize],
            None => 0,
        }
    }

    fn reward_quiet(&mut self, side: Color, best: Move, tried: &[Move], depth: i32, ply: usize) {
        if self.killers[ply][0] != Some(best) {
            self.killers[ply][1] = self.killers[ply][0];
            self.killers[ply][0] = Some(best);
        }
        let bonus = (depth * depth).min(400);
        self.update_history(side, best, bonus);
        for &m in tried {
            self.update_history(side, m, -bonus);
        }
    }

    fn update_history(&mut self, side: Color, m: Move, bonus: i32) {
        if let Some(from) = m.from() {
            let entry = &mut self.history[color_index(side)][from as usize][m.to() as usize];
            *entry += bonus - *entry * bonus.abs() / HISTORY_LIMIT;
        }
    }

    fn extract_pv(&self, pos: &Chess, first: Move) -> Vec<Move> {
        let mut pv = vec![first];
        let mut current = pos.clone();
        current.play_unchecked(first);
        let mut seen = vec![position_key(pos), position_key(&current)];
        while pv.len() < 32 {
            let key = position_key(&current);
            let Some(m) = self.tt.probe(key).and_then(|entry| entry.mv) else { break };
            if !current.is_legal(m) {
                break;
            }
            current.play_unchecked(m);
            let next = position_key(&current);
            if seen.contains(&next) {
                break;
            }
            seen.push(next);
            pv.push(m);
        }
        pv
    }

    fn print_iteration(&self, pos: &Chess, depth: i32, score: i32) {
        let millis = self.start.elapsed().as_millis().max(1) as u64;
        let nps = self.nodes * 1000 / millis;
        let pv = self
            .root_best
            .map(|m| self.extract_pv(pos, m))
            .unwrap_or_default()
            .iter()
            .map(|m| m.to_uci(CastlingMode::Standard).to_string())
            .collect::<Vec<_>>()
            .join(" ");
        println!(
            "info depth {depth} seldepth {} score {} nodes {} nps {nps} time {millis} pv {pv}",
            self.seldepth,
            format_score(score),
            self.nodes,
        );
    }
}
