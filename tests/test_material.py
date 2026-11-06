import chess

from illusion.material import see


def test_queen_takes_defended_pawn_loses_material():
    board = chess.Board("4k3/8/3p4/4p3/8/8/4Q3/4K3 w - - 0 1")
    assert see(board, chess.Move.from_uci("e2e5")) == 100 - 900


def test_queen_takes_undefended_pawn_wins_a_pawn():
    board = chess.Board("4k3/8/8/4p3/8/8/4Q3/4K3 w - - 0 1")
    assert see(board, chess.Move.from_uci("e2e5")) == 100


def test_quiet_safe_move_is_zero():
    assert see(chess.Board(), chess.Move.from_uci("e2e4")) == 0


def test_even_trade_is_zero():
    board = chess.Board("4k3/8/5n2/3n4/8/2N5/8/4K3 w - - 0 1")
    assert see(board, chess.Move.from_uci("c3d5")) == 0


def test_en_passant_counts_the_captured_pawn():
    board = chess.Board("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1")
    assert see(board, chess.Move.from_uci("e5d6")) == 100
