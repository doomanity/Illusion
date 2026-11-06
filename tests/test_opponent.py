import math
import random

import chess
import pytest

from illusion import EloModel, UniformModel

QUEEN_TRAP = "rnb1kbnr/pppp1ppp/8/4p3/4P2q/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 1"
TAKE_QUEEN = chess.Move.from_uci("f3h4")


@pytest.mark.parametrize("elo", [600, 1200, 2400])
def test_distribution_sums_to_one_and_covers_all_moves(elo):
    board = chess.Board()
    dist = EloModel(elo).distribution(board)
    assert math.isclose(sum(p for _, p in dist), 1.0)
    assert {m for m, _ in dist} == set(board.legal_moves)


def test_distribution_is_sorted_best_first():
    probs = [p for _, p in EloModel(1500).distribution(chess.Board(QUEEN_TRAP))]
    assert probs == sorted(probs, reverse=True)


def test_stronger_players_find_the_free_queen_more_often():
    board = chess.Board(QUEEN_TRAP)
    probability = lambda elo: dict(EloModel(elo).distribution(board))[TAKE_QUEEN]
    assert probability(2400) > probability(800)
    assert probability(2400) > 0.5


def test_predict_returns_top_k():
    assert len(EloModel(1200, top_k=3).predict(chess.Board())) == 3


def test_sampling_is_reproducible_with_a_seeded_rng():
    board = chess.Board(QUEEN_TRAP)
    model = EloModel(1000)
    assert model.sample(board, random.Random(5)) == model.sample(board, random.Random(5))


def test_sampled_move_is_legal():
    board = chess.Board(QUEEN_TRAP)
    assert EloModel(800).sample(board, random.Random(1)) in board.legal_moves


def test_no_moves_gives_empty_distribution():
    assert EloModel().distribution(chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")) == []


def test_uniform_model():
    dist = UniformModel().predict(chess.Board())
    assert len(dist) == 20 and math.isclose(sum(p for _, p in dist), 1.0)
