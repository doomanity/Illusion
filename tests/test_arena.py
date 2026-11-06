import random

import chess
import chess.pgn

import arena


def test_play_game_respects_the_ply_limit(monkeypatch):
    monkeypatch.setattr(arena, "MAX_PLIES", 6)
    record = arena.play_game(depth=1, opponent_elo=1000, illusion_is_white=True, rng=random.Random(0))
    assert len(record.board.move_stack) == 6
    assert record.illusion_score in (0.0, 0.5, 1.0)


def test_score_is_from_illusions_point_of_view():
    board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
    stalemate = arena.GameRecord(board, illusion_is_white=True, traps=[])
    assert stalemate.illusion_score == 0.5

    mated = chess.Board("6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1")
    mated.push_san("Re8#")
    as_white = arena.GameRecord(mated, illusion_is_white=True, traps=[])
    as_black = arena.GameRecord(mated, illusion_is_white=False, traps=[])
    assert as_white.illusion_score == 1.0 and as_black.illusion_score == 0.0


def test_pgn_names_both_players():
    board = chess.Board()
    board.push_san("e4")
    game = arena.to_pgn(arena.GameRecord(board, illusion_is_white=False, traps=[]), 1200)
    assert game.headers["Black"] == "Illusion"
    assert game.headers["White"] == "SimHuman1200"
