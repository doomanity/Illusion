"""Static position evaluation. Scores are centipawns from the side to move's view."""
from __future__ import annotations

import chess

PIECE_VALUES: dict[chess.PieceType, int] = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}

KNIGHT_CENTER_WEIGHT = 4
BISHOP_CENTER_WEIGHT = 2
PAWN_ADVANCE_WEIGHT = 3


def centrality(square: chess.Square) -> int:
    """5 on the four central squares, -1 in the corners."""
    file_distance = abs(chess.square_file(square) - 3.5)
    rank_distance = abs(chess.square_rank(square) - 3.5)
    return int(6 - (file_distance + rank_distance))


def _relative_rank(square: chess.Square, color: chess.Color) -> int:
    rank = chess.square_rank(square)
    return rank if color == chess.WHITE else 7 - rank


def _piece_score(piece: chess.Piece, square: chess.Square) -> int:
    score = PIECE_VALUES[piece.piece_type]
    if piece.piece_type == chess.KNIGHT:
        score += KNIGHT_CENTER_WEIGHT * centrality(square)
    elif piece.piece_type == chess.BISHOP:
        score += BISHOP_CENTER_WEIGHT * centrality(square)
    elif piece.piece_type == chess.PAWN:
        score += centrality(square) + PAWN_ADVANCE_WEIGHT * _relative_rank(square, piece.color)
    return score


def evaluate(board: chess.Board) -> int:
    """Material plus small positional bonuses, from the side to move's view."""
    white_minus_black = 0
    for square, piece in board.piece_map().items():
        score = _piece_score(piece, square)
        white_minus_black += score if piece.color == chess.WHITE else -score
    return white_minus_black if board.turn == chess.WHITE else -white_minus_black
