import chess

from illusion import EngineEvaluator, classify
from illusion.brilliant import Report

QUEEN_INTO_PAWN = "4k3/8/3p4/4p3/8/8/4Q3/4K3 w - - 0 1"
QXE5 = chess.Move.from_uci("e2e5")


class FlatEvaluator:
    """Every position looks equal, so any move is 'best'."""

    def value(self, board):
        return 0


def test_quiet_move_is_not_a_sacrifice():
    report = classify(chess.Board(), chess.Move.from_uci("e2e4"), FlatEvaluator())
    assert not report.is_brilliant
    assert "not a real sacrifice" in report.failures


def test_sacrifice_that_stays_best_is_brilliant():
    report = classify(chess.Board(QUEEN_INTO_PAWN), QXE5, FlatEvaluator())
    assert report.is_brilliant
    assert report.sacrifice_cp >= 200


def test_real_engine_rejects_a_plain_blunder():
    report = classify(chess.Board(QUEEN_INTO_PAWN), QXE5, EngineEvaluator(depth=2))
    assert not report.is_brilliant
    assert "not best or near best" in report.failures


def test_already_winning_positions_are_not_brilliant():
    class Crushing:
        def value(self, board):
            return -900  # opponent is lost after any move, so we are winning

    report = classify(chess.Board(QUEEN_INTO_PAWN), QXE5, Crushing())
    assert "already winning big" in report.failures


def test_leaving_yourself_worse_is_rejected():
    class Bad:
        def value(self, board):
            return 100  # the opponent is always better after our move

    report = classify(chess.Board(QUEEN_INTO_PAWN), QXE5, Bad())
    assert "leaves you worse" in report.failures


def test_report_is_immutable_and_board_untouched():
    board = chess.Board(QUEEN_INTO_PAWN)
    report = classify(board, QXE5, FlatEvaluator())
    assert isinstance(report, Report)
    assert board.fen() == QUEEN_INTO_PAWN and not board.move_stack
