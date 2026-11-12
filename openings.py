"""Find your own trappy openings (Alien Gambit style) by searching against human players.

Quick test with the built-in pieces (weak, but no installs needed):
    python openings.py --side white --my-moves 2

The real thing (stockfish judges, maia3 plays the humans):
    python openings.py --side white --stockfish stockfish.exe --maia3 23m --opp-elo 1600 --my-elo 1600
    python openings.py --side black --stockfish stockfish.exe --maia3 23m --opp-elo 1600 --my-elo 1600

Grow an opening you already like, e.g. keep 1.e4 and go deeper:
    python openings.py --side white --start "e4 c6 d4 d5" --my-moves 5 ...

Outputs a PGN (drop it into a lichess study to drill it) and a book file the
engine can load with `setoption name BookFile value <path>`.
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
    collect_traps,
    positions_to_learn,
    to_pgn,
)

VERIFY_PLIES = 80  # verification games stop here and the engine decides the result
VERIFY_DECISIVE = 300  # centipawns that count as a won position at the cut-off


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
    parser.add_argument("--verify", type=int, default=0, help="play this many test games after the search")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--pgn", default=None, help="output PGN (default: opening_<side>.pgn)")
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


def verify(tree: Node, start: chess.Board, judge, opponent, config: FinderConfig, games: int, rng: random.Random):
    """Play test games: you follow the repertoire, then the engine plays on; the human model plays them."""
    book = our_moves(tree, start.copy())
    wins = draws = losses = 0
    for _ in range(games):
        board = start.copy()
        while not board.is_game_over(claim_draw=True) and len(board.move_stack) < VERIFY_PLIES:
            if board.turn == config.side:
                move = book.get(board.epd()) or engine_move(judge, board, config.judge_depth)
            else:
                move = sample_move(opponent.distribution(board), rng)
            board.push(move)
        outcome = board.outcome(claim_draw=True)
        if outcome is not None:
            score = 0.5 if outcome.winner is None else float(outcome.winner == config.side)
        else:
            cp = judge.value(board, config.judge_depth)
            cp = cp if board.turn == config.side else -cp
            score = 1.0 if cp >= VERIFY_DECISIVE else 0.0 if cp <= -VERIFY_DECISIVE else 0.5
        wins += score == 1.0
        draws += score == 0.5
        losses += score == 0.0
    return wins, draws, losses


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = build_config(args)
    start = start_board(args.start)
    name = args.name or f"Illusion {args.side.title()} Weapon"
    pgn_path = args.pgn or f"opening_{args.side}.pgn"
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

        print(f"\nexpected score vs {args.opp_elo}s: {tree.value:.0%}")
        print(f"moves to memorise: {positions_to_learn(tree)}")
        traps = list(collect_traps(tree, start.copy(), config))
        print(f"traps found: {len(traps)}")
        for board, reply, best in sorted(traps, key=lambda t: -t[1].probability * t[1].loss)[:10]:
            line = chess.Board().variation_san(board.move_stack) if board.move_stack else "(start)"
            fix = board.san(best) if best else "?"
            print(f"  {line} ... {board.san(reply.move)}? played {reply.probability:.0%}, "
                  f"loses {reply.loss / 100:.1f} (only good move: {fix})")

        with open(pgn_path, "w", encoding="utf-8") as handle:
            print(to_pgn(tree, start, config, name, args.opp_elo), file=handle)
        with open(book_path, "w", encoding="utf-8") as handle:
            handle.write(book_text(tree, start, config, name))
        print(f"\nsaved {pgn_path} (open in a lichess study) and {book_path} (engine BookFile)")

        if args.verify:
            wins, draws, losses = verify(tree, start, judge, opponent, config, args.verify, random.Random(args.seed))
            print(f"test games vs {args.opp_elo}s: +{wins} ={draws} -{losses}")
    finally:
        if args.stockfish:
            judge.close()


if __name__ == "__main__":
    main()
