"""Alpha-beta negamax search with quiescence and a transposition table.

All scores are centipawns from the side to move's view.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Iterable, NamedTuple

import chess
import chess.polyglot

from .evaluate import PIECE_VALUES, evaluate

MATE = 100_000
INF = 10**9
MATE_THRESHOLD = MATE - 1000  # scores beyond this mean "a forced mate was found"
MAX_CHECK_EXTENSION_PLY = 20

TT_MOVE_PRIORITY = 10_000
CAPTURE_PRIORITY = 1000
PROMOTION_PRIORITY = 900
CHECK_PRIORITY = 50


class Bound(IntEnum):
    EXACT = 0
    LOWER = 1
    UPPER = 2


class TTEntry(NamedTuple):
    depth: int
    bound: Bound
    score: int
    move: chess.Move | None


@dataclass(frozen=True)
class SearchResult:
    move: chess.Move | None
    score: int
    depth: int
    nodes: int


def is_mate_score(score: int) -> bool:
    return abs(score) >= MATE_THRESHOLD


def _move_priority(board: chess.Board, move: chess.Move) -> int:
    priority = 0
    if board.is_capture(move):
        victim = board.piece_type_at(move.to_square) or chess.PAWN  # en passant
        attacker = board.piece_type_at(move.from_square)
        priority += CAPTURE_PRIORITY + 10 * PIECE_VALUES[victim] - PIECE_VALUES[attacker] // 10
    if move.promotion:
        priority += PROMOTION_PRIORITY
    if board.gives_check(move):
        priority += CHECK_PRIORITY
    return priority


def order_moves(
    board: chess.Board, moves: Iterable[chess.Move], preferred: chess.Move | None = None
) -> list[chess.Move]:
    """Best guesses first so alpha-beta can cut more: hash move, captures (MVV-LVA), checks."""

    def key(move: chess.Move) -> int:
        return TT_MOVE_PRIORITY if move == preferred else _move_priority(board, move)

    return sorted(moves, key=key, reverse=True)


def _is_draw(board: chess.Board) -> bool:
    return (
        board.is_stalemate()
        or board.is_insufficient_material()
        or board.can_claim_fifty_moves()
        or board.is_repetition(2)
    )


class Engine:
    def __init__(self, depth: int = 4):
        self.depth = depth
        self.nodes = 0
        self._table: dict[int, TTEntry] = {}

    # ---- public API -------------------------------------------------------
    def search(self, board: chess.Board, depth: int | None = None) -> SearchResult:
        """Iterative deepening from the root; returns the best move found."""
        depth = depth or self.depth
        board = board.copy()
        self.nodes = 0
        moves = list(board.legal_moves)
        if not moves:
            return SearchResult(None, -MATE if board.is_checkmate() else 0, 0, 0)

        best_move, best_score = moves[0], -INF
        for current_depth in range(1, depth + 1):
            best_move, best_score = self._search_root(board, moves, current_depth, best_move)
        return SearchResult(best_move, best_score, depth, self.nodes)

    def value(self, board: chess.Board, depth: int) -> int:
        """Exact minimax value of a position (full window)."""
        return self._negamax(board.copy(), depth, -INF, INF, 0)

    def move_value(self, board: chess.Board, move: chess.Move, depth: int) -> int:
        """Exact value of playing `move`, from the mover's view."""
        after = board.copy()
        after.push(move)
        return -self._negamax(after, depth - 1, -INF, INF, 1)

    # ---- internals --------------------------------------------------------
    def _search_root(
        self, board: chess.Board, moves: list[chess.Move], depth: int, previous_best: chess.Move
    ) -> tuple[chess.Move, int]:
        alpha = -INF
        best_move, best_score = previous_best, -INF
        for move in order_moves(board, moves, previous_best):
            board.push(move)
            score = -self._negamax(board, depth - 1, -INF, -alpha, 1)
            board.pop()
            if score > best_score:
                best_move, best_score = move, score
            alpha = max(alpha, score)
        return best_move, best_score

    def _negamax(self, board: chess.Board, depth: int, alpha: int, beta: int, ply: int) -> int:
        self.nodes += 1
        moves = list(board.legal_moves)
        if not moves:
            return -MATE + ply if board.is_checkmate() else 0
        if _is_draw(board):
            return 0
        if board.is_check() and ply < MAX_CHECK_EXTENSION_PLY:
            depth += 1
        if depth <= 0:
            return self._quiesce(board, alpha, beta)

        key = chess.polyglot.zobrist_hash(board)
        entry = self._table.get(key)
        if entry and entry.depth >= depth:
            alpha, beta, cutoff = _narrow_window(entry, alpha, beta)
            if cutoff:
                return entry.score

        original_alpha = alpha
        best_score, best_move = -INF, None
        for move in order_moves(board, moves, entry.move if entry else None):
            board.push(move)
            score = -self._negamax(board, depth - 1, -beta, -alpha, ply + 1)
            board.pop()
            if score > best_score:
                best_score, best_move = score, move
            alpha = max(alpha, score)
            if alpha >= beta:
                break

        if not is_mate_score(best_score):  # mate scores depend on ply, so never cache them
            bound = (
                Bound.UPPER if best_score <= original_alpha
                else Bound.LOWER if best_score >= beta
                else Bound.EXACT
            )
            self._table[key] = TTEntry(depth, bound, best_score, best_move)
        return best_score

    def _quiesce(self, board: chess.Board, alpha: int, beta: int) -> int:
        """Keep resolving captures so we never evaluate in the middle of a trade."""
        self.nodes += 1
        stand_pat = evaluate(board)
        if stand_pat >= beta:
            return stand_pat
        alpha = max(alpha, stand_pat)
        for move in order_moves(board, board.generate_legal_captures()):
            board.push(move)
            score = -self._quiesce(board, -beta, -alpha)
            board.pop()
            if score >= beta:
                return score
            alpha = max(alpha, score)
        return alpha


def _narrow_window(entry: TTEntry, alpha: int, beta: int) -> tuple[int, int, bool]:
    """Use a cached bound to tighten the window. The flag says a cutoff is possible."""
    if entry.bound == Bound.EXACT:
        return alpha, beta, True
    if entry.bound == Bound.LOWER:
        alpha = max(alpha, entry.score)
    else:
        beta = min(beta, entry.score)
    return alpha, beta, alpha >= beta
