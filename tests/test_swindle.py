import chess

from illusion import EloModel, Engine, swindle_choose
from illusion.search import MATE

E4, D4, KNIGHT_SIDESTEP = (chess.Move.from_uci(u) for u in ("e2e4", "d2d4", "g1h3"))
E5 = chess.Move.from_uci("e7e5")


class ScriptedEngine:
    """Every move looks equal to a perfect opponent, except Nh3 which is bad."""

    def move_value(self, board, move, depth):
        return -500 if move == KNIGHT_SIDESTEP else 0

    def value(self, board, depth):
        trapped = board.move_stack[-2:] == [E4, E5]
        sidestep = board.move_stack[-2] == KNIGHT_SIDESTEP
        return 300 if trapped else 2000 if sidestep else 0


class ScriptedModel:
    def predict(self, board):
        return [(E5, 1.0)] if board.peek() == E4 else [(chess.Move.from_uci("d7d5"), 1.0)]


def test_picks_the_move_the_model_falls_for():
    choice = swindle_choose(chess.Board(), ScriptedEngine(), ScriptedModel(), depth=3)
    assert choice.move == E4
    assert choice.trap_gain == 300


def test_never_picks_a_move_outside_the_tolerance():
    choice = swindle_choose(chess.Board(), ScriptedEngine(), ScriptedModel(), depth=3, tolerance=60)
    assert choice.move != KNIGHT_SIDESTEP  # model "expects" +2000 there, but it is objectively bad


def test_zero_tolerance_falls_back_to_best_move():
    choice = swindle_choose(chess.Board(), ScriptedEngine(), ScriptedModel(), depth=3, tolerance=0)
    assert choice.minimax == 0


def test_takes_a_forced_mate_immediately():
    board = chess.Board("6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1")
    choice = swindle_choose(board, Engine(2), ScriptedModel(), depth=2)
    assert choice.move == chess.Move.from_uci("e1e8")
    assert choice.minimax == MATE - 1
    assert choice.trap_gain == 0  # a forced mate is not a trap


def test_works_with_the_real_engine_and_leaves_the_board_untouched():
    board = chess.Board()
    choice = swindle_choose(board, Engine(2), EloModel(1000), depth=2)
    assert choice.move in board.legal_moves
    assert board.fen() == chess.STARTING_FEN and not board.move_stack
