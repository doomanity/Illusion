import random

import chess
import pytest

from illusion import Engine
from illusion.search import MATE, is_mate_score

MATE_IN_ONE = "6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1"
MATE_IN_TWO = "k7/8/8/1K6/8/8/8/7R w - - 0 1"
AVOID_STALEMATE = "7k/8/6K1/8/8/8/8/5Q2 w - - 0 1"  # Qf7 would stalemate, Qf8 mates
STALEMATE = "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"
QUEEN_TRAP = "rnb1kbnr/pppp1ppp/8/4p3/4P2q/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 1"
TT_POSITIONS = [chess.STARTING_FEN, QUEEN_TRAP, MATE_IN_TWO]


def test_finds_mate_in_one():
    result = Engine(2).search(chess.Board(MATE_IN_ONE))
    assert result.move == chess.Move.from_uci("e1e8")
    assert result.score == MATE - 1


def test_finds_mate_in_two():
    result = Engine(4).search(chess.Board(MATE_IN_TWO))
    assert is_mate_score(result.score)
    assert result.score >= MATE - 5


def test_avoids_stalemating_when_winning():
    result = Engine(2).search(chess.Board(AVOID_STALEMATE))
    assert result.move != chess.Move.from_uci("f1f7")
    assert is_mate_score(result.score)


def test_stalemate_is_a_draw_with_no_move():
    result = Engine(2).search(chess.Board(STALEMATE))
    assert result.move is None and result.score == 0


def test_takes_the_free_queen():
    result = Engine(2).search(chess.Board(QUEEN_TRAP))
    assert result.move == chess.Move.from_uci("f3h4")


def test_search_score_matches_best_move_value():
    board = chess.Board(QUEEN_TRAP)
    engine = Engine(3)
    result = engine.search(board)
    assert engine.move_value(board, result.move, 3) == result.score


@pytest.mark.parametrize("fen", TT_POSITIONS)
def test_warm_table_gives_same_value_as_fresh_engine(fen):
    board = chess.Board(fen)
    warm = Engine(3)
    for other in TT_POSITIONS:
        warm.search(chess.Board(other), depth=3)
    assert warm.value(board, 3) == Engine(3).value(board, 3)


def test_search_never_returns_illegal_moves_in_random_positions():
    rng = random.Random(7)
    for _ in range(8):
        board = chess.Board()
        for _ in range(rng.randint(4, 30)):
            if board.is_game_over():
                break
            board.push(rng.choice(list(board.legal_moves)))
        if board.is_game_over():
            continue
        result = Engine(2).search(board)
        assert result.move in board.legal_moves


def test_search_does_not_mutate_the_board():
    board = chess.Board(QUEEN_TRAP)
    before = board.fen(), list(board.move_stack)
    Engine(3).search(board)
    assert (board.fen(), list(board.move_stack)) == before
