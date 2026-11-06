"""Static exchange evaluation, used to spot sacrifices."""
from __future__ import annotations

import chess

from .evaluate import PIECE_VALUES


def _value_on(board: chess.Board, square: chess.Square) -> int:
    return PIECE_VALUES[board.piece_type_at(square) or chess.PAWN]


def _best_recapture_gain(board: chess.Board, square: chess.Square) -> int:
    """Most material the side to move can win by capturing on `square`.

    Always uses the cheapest attacker first, and may decline to capture (never negative).
    """
    captures = [m for m in board.generate_legal_captures() if m.to_square == square]
    if not captures:
        return 0
    cheapest = min(captures, key=lambda m: PIECE_VALUES[board.piece_type_at(m.from_square)])
    captured_value = _value_on(board, square)
    board.push(cheapest)
    gain = captured_value - _best_recapture_gain(board, square)
    board.pop()
    return max(0, gain)


def _captured_value(board: chess.Board, move: chess.Move) -> int:
    if board.is_en_passant(move):
        return PIECE_VALUES[chess.PAWN]
    victim = board.piece_type_at(move.to_square)
    return PIECE_VALUES[victim] if victim else 0


def see(board: chess.Board, move: chess.Move) -> int:
    """Net material the mover wins (negative = loses) if both sides keep
    trading on the destination square."""
    captured = _captured_value(board, move)
    after = board.copy()
    after.push(move)
    return captured - _best_recapture_gain(after, move.to_square)
