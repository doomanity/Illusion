"""Human move prediction with Maia3 (https://github.com/CSSLab/maia3).

`Maia3Backend` is a thin wrapper over the maia3 library (needs `torch` and
`maia3` installed, see requirements-maia.txt). `Maia3Model` turns its move
probabilities into an `OpponentModel`. Tests use a fake backend, so nothing
here needs torch unless you build a real `Maia3Backend`.
"""
from __future__ import annotations

import random
from typing import Protocol

import chess

from .opponent import MoveProbabilities, sample_move

MAX_MOVES = 20  # maia3 reports at most this many moves per position
DEFAULT_ELO = 1500
DEFAULT_MODEL = "23m"  # 5m, 23m or 79m


class PolicyBackend(Protocol):
    def policy(self, board: chess.Board, elo: int, oppo_elo: int) -> MoveProbabilities:
        """Probability of each likely move for a player rated `elo` facing `oppo_elo`."""


def position_command(board: chess.Board) -> str:
    """UCI `position` line that replays the game, so maia3 sees the move history."""
    command = f"position fen {board.root().fen()}"
    if board.move_stack:
        command += " moves " + " ".join(move.uci() for move in board.move_stack)
    return command


class Maia3Backend:
    def __init__(self, model: str = DEFAULT_MODEL, device: str = "cpu"):
        from maia3 import uci  # imported late: pulls in torch

        argv = ["--model", model, "--device", device, "--use-uci-history"]
        if device == "cpu":
            argv.append("--no-use-amp")
        self._engine = uci.Maia3UCIEngine(uci.parse_args(argv))
        self._engine.ensure_model_loaded()
        self._engine.cmd_setoption(f"setoption name MultiPV value {MAX_MOVES}")

    def policy(self, board: chess.Board, elo: int, oppo_elo: int) -> MoveProbabilities:
        self._engine.cmd_setoption(f"setoption name SelfElo value {elo}")
        self._engine.cmd_setoption(f"setoption name OppoElo value {oppo_elo}")
        self._engine.cmd_position(position_command(board))
        _, top_moves = self._engine.score_moves()
        return [(entry["move"], float(entry["policy"])) for entry in top_moves]


class Maia3Model:
    """Opponent model backed by maia3. `elo` is the opponent's rating and
    `oppo_elo` the rating maia3 should assume for the player they face (us)."""

    def __init__(
        self,
        backend: PolicyBackend,
        elo: int = DEFAULT_ELO,
        oppo_elo: int | None = None,
        top_k: int = 6,
    ):
        self.backend = backend
        self.elo = elo
        self.oppo_elo = oppo_elo or elo
        self.top_k = top_k

    def distribution(self, board: chess.Board) -> MoveProbabilities:
        """Legal moves with probabilities that sum to 1, most likely first."""
        legal = set(board.legal_moves)
        raw = [(m, p) for m, p in self.backend.policy(board, self.elo, self.oppo_elo) if m in legal and p > 0]
        total = sum(p for _, p in raw)
        if total <= 0:
            return []
        dist = [(move, p / total) for move, p in raw]
        dist.sort(key=lambda pair: -pair[1])
        return dist

    def predict(self, board: chess.Board) -> MoveProbabilities:
        return self.distribution(board)[: self.top_k]

    def sample(self, board: chess.Board, rng: random.Random | None = None) -> chess.Move:
        return sample_move(self.distribution(board), rng)
