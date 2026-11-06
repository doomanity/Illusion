import chess
import chess.engine
import pytest

from illusion import StockfishEngine, swindle_choose
from illusion.brilliant import classify
from illusion.search import MATE

QUEEN_INTO_PAWN = "4k3/8/3p4/4p3/8/8/4Q3/4K3 w - - 0 1"


class FakeUci:
    """Mimics chess.engine.SimpleEngine. `scores` maps a FEN to a PovScore (white's view)."""

    def __init__(self, scores=None, default=chess.engine.Cp(0)):
        self.scores = scores or {}
        self.default = default
        self.analysed = []
        self.limits = []
        self.quit_called = False

    def analyse(self, board, limit):
        self.analysed.append(board.fen())
        self.limits.append(limit)
        return {"score": chess.engine.PovScore(self.scores.get(board.fen(), self.default), chess.WHITE)}

    def quit(self):
        self.quit_called = True


def test_value_is_from_the_side_to_moves_view():
    white_up = chess.engine.Cp(120)
    fake = FakeUci(default=white_up)
    engine = StockfishEngine(fake)
    assert engine.value(chess.Board()) == 120
    black_to_move = chess.Board()
    black_to_move.push_san("e4")
    assert engine.value(black_to_move) == -120


def test_mate_scores_are_converted():
    engine = StockfishEngine(FakeUci(default=chess.engine.Mate(3)))
    assert engine.value(chess.Board()) == MATE - 3


def test_move_value_is_negated_value_after_the_move():
    engine = StockfishEngine(FakeUci(default=chess.engine.Cp(-40)))  # white's view after e4
    assert engine.move_value(chess.Board(), chess.Move.from_uci("e2e4")) == -40


def test_terminal_positions_are_scored_without_calling_stockfish():
    fake = FakeUci()
    engine = StockfishEngine(fake)
    mated = chess.Board("6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1")
    mated.push_san("Re8#")
    assert engine.value(mated) == -MATE
    assert engine.move_value(chess.Board("6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1"), chess.Move.from_uci("e1e8")) == MATE
    assert engine.value(chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")) == 0
    assert fake.analysed == []


def test_repeated_positions_are_cached():
    fake = FakeUci()
    engine = StockfishEngine(fake)
    for _ in range(3):
        engine.value(chess.Board())
    assert len(fake.analysed) == 1


def test_depth_argument_is_ignored_and_configured_depth_used():
    fake = FakeUci()
    StockfishEngine(fake, depth=9).value(chess.Board(), depth=1)
    assert fake.limits[0].depth == 9


def test_context_manager_quits_the_engine():
    fake = FakeUci()
    with StockfishEngine(fake):
        pass
    assert fake.quit_called


def test_swindle_accepts_stockfish_as_the_judge():
    fake = FakeUci()
    class Uniform:
        def predict(self, board):
            moves = list(board.legal_moves)[:3]
            return [(m, 1 / 3) for m in moves]
    choice = swindle_choose(chess.Board(), StockfishEngine(fake), Uniform(), depth=3)
    assert choice.move in chess.Board().legal_moves


def test_brilliant_detector_accepts_stockfish_as_the_evaluator():
    board = chess.Board(QUEEN_INTO_PAWN)
    report = classify(board, chess.Move.from_uci("e2e5"), StockfishEngine(FakeUci()))
    assert report.is_brilliant  # flat evals: the sacrifice looks equal-best
