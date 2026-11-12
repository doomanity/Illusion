"""Find your own trappy openings (Alien Gambit style) by searching against human players.

Quick test with the built-in pieces (weak, but no installs needed):
    python openings.py --side white --my-moves 2

The real thing (stockfish judges, maia3 plays the humans):
    python openings.py --side white --stockfish stockfish.exe --maia3 23m --opp-elo 1600 --my-elo 1600
    python openings.py --side black --stockfish stockfish.exe --maia3 23m --opp-elo 1600 --my-elo 1600

Grow an opening you already like, e.g. keep 1.e4 and go deeper:
    python openings.py --side white --start "e4 c6 d4 d5" --my-moves 5 ...

Outputs three files per side:
  opening_<side>.txt  plain text cheat sheet: one line per row, traps marked (read this one)
  opening_<side>.pgn  one short game per line, import into a lichess study (a chapter each)
  book_<side>.txt     book file for the engine: `setoption name BookFile value <path>`
"""
from __future__ import annotations

import argparse
import random
import sys
import time

import chess

from illusion import EloModel, Engine, Maia3Backend, Maia3Model, StockfishEngine
from illusion.opponent import sample_move
from illusion.openings import (
    FinderConfig,
    Node,
    OpeningFinder,
    book_text,
    cheat_sheet,
    study_pgn,
    win_chance,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--side", choices=["white", "black"], default="white")
    parser.add_argument("--start", default="", help='moves to start from in SAN, e.g. "e4 c6 d4 d5"')
    parser.add_argument("--name", default=None, help="name for your opening, e.g. 'Illusion Gambit'")
    parser.add_argument("--opp-elo", type=int, default=1500, help="rating of the humans you want to beat")
    parser.add_argument("--my-elo", type=int, default=1500, help="your rating (for the novelty bonus)")
    parser.add_argument("--stockfish", metavar="PATH", help="stockfish as the judge (strongly recommended)")
    parser.add_argument("--stockfish-depth", type=int, default=12)
    parser.add_argument("--threads", type=int, default=4, help="stockfish threads")
    parser.add_argument("--maia3", metavar="MODEL", help="maia3 as the humans: 5m, 23m or 79m")
    parser.add_argument("--device", default="cpu", help="maia3 device: cpu or cuda")
    parser.add_argument("--depth", type=int, default=3, help="built-in engine depth (when no stockfish)")
    parser.add_argument("--my-moves", type=int, default=4, help="how many of your moves deep to search")
    parser.add_argument("--candidates", type=int, default=3, help="your moves tried per position")
    parser.add_argument("--coverage", type=float, default=0.75, help="share of human replies to cover")
    parser.add_argument("--max-replies", type=int, default=3, help="most human replies per position")
    parser.add_argument("--soundness", type=int, default=250, help="max centipawns worse than best you allow")
    parser.add_argument("--truth-weight", type=float, default=0.25, help="0 = pure swindle, 1 = pure engine")
    parser.add_argument("--novelty-weight", type=float, default=0.05, help="bonus for rare moves")
    parser.add_argument("--min-line", type=float, default=0.0, help="skip opponent lines humans reach less often than this, e.g. 0.04")
    parser.add_argument("--verify", type=int, default=0, help="play this many sample openings and compare with plain engine moves")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--pgn", default=None, help="output PGN (default: opening_<side>.pgn)")
    parser.add_argument("--txt", default=None, help="output cheat sheet (default: opening_<side>.txt)")
    parser.add_argument("--book", default=None, help="output book file (default: book_<side>.txt)")
    return parser.parse_args(argv)


def start_board(text: str) -> chess.Board:
    board = chess.Board()
    for token in text.replace(".", ". ").split():
        if token.endswith(".") and token[:-1].isdigit():
            continue
        board.push_san(token)
    return board


def build_config(args: argparse.Namespace) -> FinderConfig:
    return FinderConfig(
        side=chess.WHITE if args.side == "white" else chess.BLACK,
        my_moves=args.my_moves,
        candidates=args.candidates,
        coverage=args.coverage,
        max_replies=args.max_replies,
        soundness=args.soundness,
        truth_weight=args.truth_weight,
        novelty_weight=args.novelty_weight,
        min_line=args.min_line,
        judge_depth=args.depth,
    )


def build_judge(args: argparse.Namespace):
    if args.stockfish:
        return StockfishEngine.from_path(args.stockfish, depth=args.stockfish_depth, threads=args.threads, hash_mb=256)
    return Engine(args.depth)


def build_models(args: argparse.Namespace):
    """(the humans you face, humans at your own level)"""
    if args.maia3:
        backend = Maia3Backend(args.maia3, args.device)
        return (
            Maia3Model(backend, elo=args.opp_elo, oppo_elo=args.my_elo),
            Maia3Model(backend, elo=args.my_elo, oppo_elo=args.opp_elo),
        )
    return EloModel(args.opp_elo), EloModel(args.my_elo)


def our_moves(tree: Node, board: chess.Board) -> dict[str, chess.Move]:
    """Position -> the repertoire move, for following the book in test games."""
    table: dict[str, chess.Move] = {}

    def walk(node: Node) -> None:
        if node.ours:
            if node.move is not None and node.child is not None:
                table[board.epd()] = node.move
                board.push(node.move)
                walk(node.child)
                board.pop()
            return
        for reply in node.replies:
            board.push(reply.move)
            walk(reply.child)
            board.pop()

    walk(tree)
    return table


def engine_move(judge, board: chess.Board, depth: int) -> chess.Move:
    ranker = getattr(judge, "top_moves", None)
    if ranker is not None:
        return ranker(board, 1)[0][0]
    return judge.search(board, depth).move


def play_opening(book, start: chess.Board, judge, opponent, config: FinderConfig, rng: random.Random) -> float:
    """Play the opening part only: you follow `book` (then the engine's best move), the human model
    answers. Returns your win chance at the end, judged by the engine."""
    board = start.copy()
    plies = 2 * config.my_moves - (1 if config.side == chess.BLACK else 0)
    for _ in range(plies):
        if board.is_game_over(claim_draw=True):
            break
        if board.turn == config.side:
            move = book.get(board.epd()) or engine_move(judge, board, config.judge_depth)
        else:
            dist = opponent.distribution(board)
            move = sample_move(dist, rng) if dist else engine_move(judge, board, config.judge_depth)
        board.push(move)
    cp = judge.value(board, config.judge_depth)
    return win_chance(cp if board.turn == config.side else -cp)


def verify(tree: Node, start: chess.Board, judge, opponent, config: FinderConfig, games: int, rng: random.Random):
    """(your score with the repertoire, your score with plain best moves), both over the same number of
    sampled openings against the human model. The difference is what the repertoire really gives you."""
    book = our_moves(tree, start.copy())
    with_book = sum(play_opening(book, start, judge, opponent, config, rng) for _ in range(games)) / games
    plain = sum(play_opening({}, start, judge, opponent, config, rng) for _ in range(games)) / games
    return with_book, plain


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = build_config(args)
    start = start_board(args.start)
    name = args.name or f"Illusion {args.side.title()} Weapon"
    pgn_path = args.pgn or f"opening_{args.side}.pgn"
    txt_path = args.txt or f"opening_{args.side}.txt"
    book_path = args.book or f"book_{args.side}.txt"

    judge = build_judge(args)
    opponent, me = build_models(args)
    began = time.time()

    def progress(count: int) -> None:
        if count % 25 == 0:
            print(f"\r  searched {count} positions ({time.time() - began:.0f}s)", end="", file=sys.stderr, flush=True)

    print(f"searching {name}: {args.side}, {args.my_moves} moves deep, vs {args.opp_elo}s ...")
    finder = OpeningFinder(judge, opponent, config, me=me, progress=progress)
    try:
        tree = finder.find(start)
        print(f"\r  searched {finder.positions} positions in {time.time() - began:.0f}s      ", file=sys.stderr)

        sheet = cheat_sheet(tree, start, config, name, args.opp_elo)
        print("\n" + sheet)
        with open(txt_path, "w", encoding="utf-8") as handle:
            handle.write(sheet)
        with open(pgn_path, "w", encoding="utf-8") as handle:
            handle.write(study_pgn(tree, start, config, name, args.opp_elo))
        with open(book_path, "w", encoding="utf-8") as handle:
            handle.write(book_text(tree, start, config, name))
        print(f"saved {txt_path} (read it), {pgn_path} (lichess study) and {book_path} (engine BookFile)")

        if args.verify:
            with_book, plain = verify(tree, start, judge, opponent, config, args.verify, random.Random(args.seed))
            result = (f"after the opening, vs {args.opp_elo}s ({args.verify} samples): "
                      f"repertoire {with_book:.0%}, plain engine moves {plain:.0%}, edge {with_book - plain:+.1%}")
            print(result)
            with open(txt_path, "a", encoding="utf-8") as handle:
                handle.write("\n" + result + "\n")
    finally:
        if args.stockfish:
            judge.close()


if __name__ == "__main__":
    main()
