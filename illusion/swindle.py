"""Trap-setting move chooser: expectimax over a human opponent model."""
from __future__ import annotations

from dataclasses import dataclass

import chess

from .opponent import OpponentModel
from .search import Engine, is_mate_score

SCORE_CLIP = 2000  # stops mate scores from wrecking the averages
DEFAULT_TOLERANCE = 60  # how much worse than best a trap move may be, in centipawns


@dataclass(frozen=True)
class SwindleChoice:
    move: chess.Move
    expected: float  # expected score against the modelled opponent
    minimax: int  # score against a perfect opponent
    candidates: int  # how many moves were safe enough to consider

    @property
    def trap_gain(self) -> float:
        """How much better this move is against the model than against perfect play."""
        return self.expected - clip(self.minimax)


def clip(score: float) -> float:
    return max(-SCORE_CLIP, min(SCORE_CLIP, score))


def _expected_score(
    board: chess.Board, move: chess.Move, minimax: int, engine: Engine, model: OpponentModel, depth: int
) -> float:
    """Average our score over the opponent's likely replies to `move`."""
    board.push(move)
    try:
        if board.is_game_over():
            return clip(minimax)
        covered, expected = 0.0, 0.0
        for reply, probability in model.predict(board):
            board.push(reply)
            expected += probability * clip(engine.value(board, max(0, depth - 2)))
            board.pop()
            covered += probability
        return expected + (1 - covered) * clip(minimax)  # unmodelled replies: assume best defence
    finally:
        board.pop()


def choose(
    board: chess.Board,
    engine: Engine,
    model: OpponentModel,
    depth: int = 3,
    tolerance: int = DEFAULT_TOLERANCE,
) -> SwindleChoice:
    """Best expected move against `model`, among moves within `tolerance` of the objective best."""
    scores = {move: engine.move_value(board, move, depth) for move in board.legal_moves}
    best_move = max(scores, key=scores.get)
    if is_mate_score(scores[best_move]):  # forced win found, just take it
        return SwindleChoice(best_move, clip(scores[best_move]), scores[best_move], 1)

    safe_moves = [m for m, score in scores.items() if score >= scores[best_move] - tolerance]
    chosen, chosen_expected = best_move, clip(scores[best_move])
    for move in safe_moves:
        expected = _expected_score(board, move, scores[move], engine, model, depth)
        if expected > chosen_expected:
            chosen, chosen_expected = move, expected
    return SwindleChoice(chosen, chosen_expected, scores[chosen], len(safe_moves))
