"""Brilliant move detector (v0).

A move is flagged when it
  1. sacrifices material (static exchange says we lose at least MIN_SACRIFICE),
  2. is still best or near best (within MAX_LOSS of the top move),
  3. does not leave us worse than MIN_AFTER,
  4. was not played from an already crushing position (best <= MAX_BEFORE).

Limitation: only sacrifices on the destination square are seen.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import chess

from .material import see
from .search import Engine

MIN_SACRIFICE = 200
MAX_LOSS = 40
MIN_AFTER = -50
MAX_BEFORE = 300


class Evaluator(Protocol):
    def value(self, board: chess.Board) -> int:
        """Centipawns from the side to move's view."""


class EngineEvaluator:
    def __init__(self, depth: int = 3):
        self._engine = Engine(depth)
        self._depth = depth

    def value(self, board: chess.Board) -> int:
        return self._engine.value(board, self._depth)


@dataclass(frozen=True)
class Report:
    move: chess.Move
    sacrifice_cp: int
    loss_vs_best: int
    best_cp: int
    after_cp: int
    failures: tuple[str, ...]

    @property
    def is_brilliant(self) -> bool:
        return not self.failures


def _score_after_each_move(board: chess.Board, evaluator: Evaluator) -> dict[chess.Move, int]:
    """Value of every legal move, from the mover's view."""
    scores = {}
    for move in list(board.legal_moves):
        board.push(move)
        scores[move] = -evaluator.value(board)
        board.pop()
    return scores


def classify(
    board: chess.Board,
    move: chess.Move,
    evaluator: Evaluator,
    min_sacrifice: int = MIN_SACRIFICE,
    max_loss: int = MAX_LOSS,
    min_after: int = MIN_AFTER,
    max_before: int = MAX_BEFORE,
) -> Report:
    sacrifice = -see(board, move)
    scores = _score_after_each_move(board, evaluator)
    best, after = max(scores.values()), scores[move]
    loss = best - after

    failures = []
    if sacrifice < min_sacrifice:
        failures.append("not a real sacrifice")
    if loss > max_loss:
        failures.append("not best or near best")
    if after < min_after:
        failures.append("leaves you worse")
    if best > max_before:
        failures.append("already winning big")
    return Report(move, sacrifice, loss, best, after, tuple(failures))
