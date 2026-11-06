"""Opponent models: given a position, how likely is each reply?

`EloModel` is a cheap stand-in for a real human-move network (Maia). Swap it
out by implementing the same `predict` method.
"""
from __future__ import annotations

import math
import random
from typing import Protocol

import chess

from .search import Engine

MoveProbabilities = list[tuple[chess.Move, float]]

MIN_ELO_FOR_NOISE, MAX_ELO_FOR_NOISE = 600, 2400
MIN_TEMPERATURE, MAX_TEMPERATURE = 20, 400  # centipawns
LOGIT_FLOOR = -50.0  # keeps hopeless moves from underflowing

CAPTURE_BONUS = 1.0
CHECK_BONUS = 0.7
PROMOTION_BONUS = 1.0
RECAPTURE_BONUS = 1.0


class OpponentModel(Protocol):
    def predict(self, board: chess.Board) -> MoveProbabilities:
        """Most likely replies with their probabilities, best first."""


class UniformModel:
    def predict(self, board: chess.Board) -> MoveProbabilities:
        moves = list(board.legal_moves)
        return [(move, 1 / len(moves)) for move in moves]


class EloModel:
    """Weaker players are noisier and love captures, checks and recaptures."""

    def __init__(self, elo: int = 1200, top_k: int = 6, engine: Engine | None = None):
        self.elo = elo
        self.top_k = top_k
        self._engine = engine or Engine(depth=1)

    @property
    def temperature(self) -> float:
        """How much a player's choice blurs between moves. Higher = weaker."""
        weakness = (MAX_ELO_FOR_NOISE - self.elo) / (MAX_ELO_FOR_NOISE - MIN_ELO_FOR_NOISE)
        weakness = min(1.0, max(0.0, weakness))
        return MIN_TEMPERATURE + (MAX_TEMPERATURE - MIN_TEMPERATURE) * weakness

    @property
    def impulsiveness(self) -> float:
        return 1.5 - self.elo / MAX_ELO_FOR_NOISE

    def distribution(self, board: chess.Board) -> MoveProbabilities:
        """Probability of every legal move, most likely first."""
        moves = list(board.legal_moves)
        if not moves:
            return []
        logits = [self._logit(board, move) for move in moves]
        peak = max(logits)
        weights = [math.exp(logit - peak) for logit in logits]
        total = sum(weights)
        dist = [(move, weight / total) for move, weight in zip(moves, weights)]
        dist.sort(key=lambda pair: -pair[1])
        return dist

    def predict(self, board: chess.Board) -> MoveProbabilities:
        return self.distribution(board)[: self.top_k]

    def sample(self, board: chess.Board, rng: random.Random | None = None) -> chess.Move:
        rng = rng or random.Random()
        dist = self.distribution(board)
        return rng.choices([m for m, _ in dist], weights=[p for _, p in dist])[0]

    def _logit(self, board: chess.Board, move: chess.Move) -> float:
        quality = self._engine.move_value(board, move, 1) / self.temperature
        return max(LOGIT_FLOOR, quality) + self._instinct_bonus(board, move) * self.impulsiveness

    @staticmethod
    def _instinct_bonus(board: chess.Board, move: chess.Move) -> float:
        bonus = 0.0
        if board.is_capture(move):
            bonus += CAPTURE_BONUS
        if board.gives_check(move):
            bonus += CHECK_BONUS
        if move.promotion:
            bonus += PROMOTION_BONUS
        if board.move_stack and move.to_square == board.peek().to_square:
            bonus += RECAPTURE_BONUS
        return bonus
