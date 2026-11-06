import math
import random

import chess
import pytest

import arena
from illusion import Engine, Maia3Model, swindle_choose
from illusion.maia import position_command


class FakeBackend:
    """Stands in for maia3: uniform over legal moves, remembers how it was asked."""

    def __init__(self, table=None):
        self.table = table
        self.calls = []

    def policy(self, board, elo, oppo_elo):
        self.calls.append((elo, oppo_elo))
        if self.table is not None:
            return self.table(board)
        moves = list(board.legal_moves)
        return [(m, 1 / len(moves)) for m in moves]


def test_position_command_from_the_start():
    assert position_command(chess.Board()) == f"position fen {chess.STARTING_FEN}"


def test_position_command_replays_the_moves():
    board = chess.Board()
    board.push_san("e4")
    board.push_san("e5")
    assert position_command(board) == f"position fen {chess.STARTING_FEN} moves e2e4 e7e5"


def test_position_command_uses_the_games_root_not_the_current_position():
    custom = "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1"
    board = chess.Board(custom)
    board.push_san("e4")
    assert position_command(board) == f"position fen {custom} moves e2e4"


def test_probabilities_are_normalised_and_sorted():
    board = chess.Board()
    e4, d4 = chess.Move.from_uci("e2e4"), chess.Move.from_uci("d2d4")
    model = Maia3Model(FakeBackend(lambda b: [(d4, 0.1), (e4, 0.3)]))
    dist = model.distribution(board)
    assert [m for m, _ in dist] == [e4, d4]
    assert math.isclose(sum(p for _, p in dist), 1.0)
    assert math.isclose(dist[0][1], 0.75)


def test_illegal_and_zero_probability_moves_are_dropped():
    board = chess.Board()
    legal, illegal = chess.Move.from_uci("e2e4"), chess.Move.from_uci("e2e5")
    zero = chess.Move.from_uci("d2d4")
    model = Maia3Model(FakeBackend(lambda b: [(legal, 0.5), (illegal, 0.5), (zero, 0.0)]))
    assert model.distribution(board) == [(legal, 1.0)]


def test_empty_policy_gives_empty_distribution():
    assert Maia3Model(FakeBackend(lambda b: [])).distribution(chess.Board()) == []


def test_predict_returns_top_k():
    assert len(Maia3Model(FakeBackend(), top_k=3).predict(chess.Board())) == 3


def test_ratings_are_passed_to_the_backend():
    backend = FakeBackend()
    Maia3Model(backend, elo=1200, oppo_elo=2400).predict(chess.Board())
    Maia3Model(backend, elo=1800).predict(chess.Board())
    assert backend.calls == [(1200, 2400), (1800, 1800)]


def test_sampling_is_reproducible_and_legal():
    board = chess.Board()
    model = Maia3Model(FakeBackend())
    first, second = model.sample(board, random.Random(3)), model.sample(board, random.Random(3))
    assert first == second and first in board.legal_moves


def test_swindle_runs_on_top_of_maia():
    board = chess.Board()
    choice = swindle_choose(board, Engine(2), Maia3Model(FakeBackend()), depth=2)
    assert choice.move in board.legal_moves


def test_arena_game_with_maia_style_players(monkeypatch):
    monkeypatch.setattr(arena, "MAX_PLIES", 6)
    belief, reality = Maia3Model(FakeBackend()), Maia3Model(FakeBackend())
    record = arena.play_game(Engine(1), belief, 1, True, random.Random(0), opponent=reality)
    assert len(record.board.move_stack) == 6
    assert len(belief.backend.calls) > 0 and len(reality.backend.calls) > 0
