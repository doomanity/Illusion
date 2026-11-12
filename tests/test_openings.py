import io
import random

import chess
import chess.pgn

import openings as cli
from illusion import EloModel, Engine
from illusion.openings import (
    FinderConfig,
    OpeningFinder,
    book_text,
    cheat_sheet,
    study_pgn,
    collect_traps,
    lines,
    positions_to_learn,
    to_pgn,
    win_chance,
)

M = chess.Move.from_uci
SOLID, GAMBIT, AWFUL = M("d2d4"), M("e2e4"), M("g1h3")
BLUNDER, ONLY_MOVE = M("e7e5"), M("d7d5")


class GambitJudge:
    """1.e4 is a bit worse than 1.d4 for an engine, but after 1...e5?? white is winning.
    1.Nh3 is objectively awful."""

    def value(self, board, depth):
        stack = board.move_stack
        if stack[:2] == [GAMBIT, BLUNDER]:
            score = 600  # white wins
        elif stack[:1] == [GAMBIT]:
            score = -100
        elif stack[:1] == [AWFUL]:
            score = -900
        else:
            score = 0
        return score if board.turn == chess.WHITE else -score

    def move_value(self, board, move, depth):
        board.push(move)
        try:
            return -self.value(board, depth)
        finally:
            board.pop()

    def top_moves(self, board, count=40):
        ranked = [(m, self.move_value(board, m, 0)) for m in board.legal_moves]
        return sorted(ranked, key=lambda pair: -pair[1])[:count]


class ScriptedHumans:
    """Humans fall for the trap 80% of the time and love answering 1.Nh3 with a blunder."""

    def distribution(self, board):
        legal = list(board.legal_moves)
        if board.move_stack[:1] == [GAMBIT] and len(board.move_stack) == 1:
            return [(BLUNDER, 0.8), (ONLY_MOVE, 0.2)]
        if board.move_stack[:1] == [AWFUL] and len(board.move_stack) == 1:
            return [(BLUNDER, 1.0)]
        return [(legal[0], 1.0)]


def config(**kw):
    defaults = dict(side=chess.WHITE, my_moves=1, candidates=40, coverage=1.0, max_replies=3)
    defaults.update(kw)
    return FinderConfig(**defaults)


def test_win_chance_is_a_probability():
    assert win_chance(0) == 0.5
    assert win_chance(400) > 0.75 and win_chance(-400) < 0.25
    assert 0 < win_chance(-10**6) < win_chance(10**6) < 1


def test_finds_the_gambit_humans_fall_for():
    tree = OpeningFinder(GambitJudge(), ScriptedHumans(), config(soundness=250)).find()
    assert tree.move == GAMBIT
    assert tree.engine_cp == -100  # the engine still thinks it is a little worse


def test_soundness_keeps_out_moves_the_engine_hates():
    tree = OpeningFinder(GambitJudge(), ScriptedHumans(), config(soundness=250)).find()
    assert tree.move != AWFUL  # humans "always" blunder there, but it is objectively losing


def test_pure_engine_weight_plays_the_solid_move():
    tree = OpeningFinder(GambitJudge(), ScriptedHumans(), config(truth_weight=1.0)).find()
    assert tree.move != GAMBIT


def test_trap_is_reported_with_the_only_move():
    cfg = config()
    tree = OpeningFinder(GambitJudge(), ScriptedHumans(), cfg).find()
    traps = list(collect_traps(tree, chess.Board(), cfg))
    assert len(traps) == 1
    board, reply, best = traps[0]
    assert board.move_stack == [GAMBIT]
    assert reply.move == BLUNDER and reply.probability == 0.8 and reply.loss == 700
    assert best != BLUNDER


def test_unexpanded_replies_are_assumed_to_be_best_defence():
    cfg = config(coverage=0.5, max_replies=1)
    tree = OpeningFinder(GambitJudge(), ScriptedHumans(), cfg).find()
    after_gambit = tree.child
    assert len(after_gambit.replies) == 1
    assert abs(after_gambit.unexpanded - 0.2) < 1e-9


def test_black_repertoire_starts_with_their_move():
    cfg = config(side=chess.BLACK, max_replies=1)
    tree = OpeningFinder(GambitJudge(), ScriptedHumans(), cfg).find()
    assert not tree.ours
    assert tree.replies and tree.replies[0].child.ours


def test_book_marks_their_moves_as_pass_through():
    cfg = config()
    tree = OpeningFinder(GambitJudge(), ScriptedHumans(), cfg).find()
    text = book_text(tree, chess.Board(), cfg, "Test Gambit")
    body = [line for line in text.splitlines() if not line.startswith("#")]
    assert sorted(body) == ["e4 ~d5", "e4 ~e5"]


def test_book_keeps_the_start_line():
    start = cli.start_board("1. Nf3 Nf6")
    cfg = config()
    tree = OpeningFinder(Engine(1), EloModel(1200), config(candidates=1, max_replies=1)).find(start)
    body = [line for line in book_text(tree, start, cfg, "T").splitlines() if not line.startswith("#")]
    assert body and all(line.startswith("Nf3 ~Nf6 ") for line in body)


def test_pgn_has_every_line_and_flags_the_trap():
    cfg = config()
    tree = OpeningFinder(GambitJudge(), ScriptedHumans(), cfg).find()
    text = str(to_pgn(tree, chess.Board(), cfg, "Test Gambit", 1500))
    assert "1. e4" in text and "e5 $4" in text and "TRAP" in text and "d5" in text


def test_memorise_count_and_lines_agree():
    cfg = config(my_moves=2, candidates=2, max_replies=2)
    tree = OpeningFinder(Engine(1), EloModel(1200), cfg).find()
    paths = lines(tree, chess.Board())
    assert len(paths) == 4  # 2 replies after each of 2 of their moves
    assert positions_to_learn(tree) == 3  # first move plus one reply move per branch


def test_verify_compares_the_repertoire_with_plain_engine_moves():
    cfg = config()
    tree = OpeningFinder(GambitJudge(), ScriptedHumans(), cfg).find()
    # the gambit wins 80% of the time against these humans, plain best (1.d4/other) just scores 50%
    with_book, plain = cli.verify(tree, chess.Board(), GambitJudge(), ScriptedHumans(), cfg, 40, random.Random(0))
    assert with_book > plain


def test_cheat_sheet_is_plain_readable_text_with_the_trap():
    cfg = config()
    tree = OpeningFinder(GambitJudge(), ScriptedHumans(), cfg).find()
    text = cheat_sheet(tree, chess.Board(), cfg, "Test Gambit", 1500)
    lines = [l for l in text.splitlines() if l[:1].isdigit()]
    assert "TEST GAMBIT" in text and lines and all("(" not in l for l in lines)  # no nested brackets
    assert "1. [80%] 1.e4 e5?" in text
    assert "after 1.e4:" in text and "humans play e5? 80%" in text and "the only good move is" in text


def test_cheat_sheet_numbers_black_moves_after_a_start_line():
    start = cli.start_board("1.e4")
    cfg = config(side=chess.BLACK, max_replies=1)
    tree = OpeningFinder(Engine(1), EloModel(1200), cfg).find(start)
    text = cheat_sheet(tree, start, cfg, "Black", 1200)
    line = next(l for l in text.splitlines() if l.startswith("1. ["))
    assert "1.e4 " in line and " 1..." not in line  # black's first reply is written plainly


def test_study_pgn_is_one_flat_game_per_line():
    cfg = config()
    tree = OpeningFinder(GambitJudge(), ScriptedHumans(), cfg).find()
    text = study_pgn(tree, chess.Board(), cfg, "Test Gambit", 1500)
    games = list(iter(lambda g=io.StringIO(text): chess.pgn.read_game(g), None))
    assert len(games) == 2
    assert all("(" not in str(g.mainline_moves()) for g in games)
    assert "TRAP" in text and games[0].headers["Event"] == "Test Gambit 1"


def test_min_line_drops_rare_opponent_lines():
    everything = OpeningFinder(GambitJudge(), ScriptedHumans(), config()).find()
    pruned = OpeningFinder(GambitJudge(), ScriptedHumans(), config(min_line=0.5)).find()
    assert len(everything.child.replies) == 2
    assert [r.move for r in pruned.child.replies] == [BLUNDER]  # the 20% reply is below the cut
    assert abs(pruned.child.unexpanded - 0.2) < 1e-9


def test_cli_flags_and_start_line():
    args = cli.parse_args(["--side", "black", "--start", "1.e4 c5", "--opp-elo", "1700", "--my-moves", "6"])
    cfg = cli.build_config(args)
    assert cfg.side == chess.BLACK and cfg.my_moves == 6 and args.opp_elo == 1700
    assert cli.start_board(args.start).move_stack == [M("e2e4"), M("c7c5")]


def test_cli_end_to_end_writes_both_files(tmp_path):
    pgn, book = tmp_path / "o.pgn", tmp_path / "b.txt"
    cli.main(["--my-moves", "1", "--depth", "1", "--max-replies", "1", "--pgn", str(pgn), "--book", str(book)])
    assert pgn.read_text().startswith("[Event")
    assert any(not line.startswith("#") for line in book.read_text().splitlines())
