"""Play Illusion against a simulated human of a given elo.

    python arena.py --games 2 --opp-elo 1200 --depth 3 --annotate

Stronger setup (needs stockfish and maia3 installed, see requirements-maia.txt):
    python arena.py --stockfish stockfish --maia3 23m --opp-elo 1500 --annotate
"""
from __future__ import annotations

import argparse
import random
from dataclasses import dataclass

import chess
import chess.pgn

from illusion import EloModel, Engine, EngineEvaluator, classify, swindle_choose
from illusion.maia import Maia3Backend, Maia3Model
from illusion.stockfish import StockfishEngine

MAX_PLIES = 160
TRAP_REPORT_THRESHOLD = 50  # centipawns of trap gain worth mentioning


@dataclass
class GameRecord:
    board: chess.Board
    illusion_is_white: bool
    traps: list[tuple[int, str, int]]  # (ply, san, trap gain)

    @property
    def result(self) -> str:
        return self.board.result(claim_draw=True)

    @property
    def illusion_score(self) -> float:
        wins_for_white = {"1-0": 1.0, "0-1": 0.0}
        white_score = wins_for_white.get(self.result, 0.5)
        return white_score if self.illusion_is_white else 1.0 - white_score


def play_game(
    engine,
    model,
    depth: int,
    illusion_is_white: bool,
    rng: random.Random,
    opponent=None,
) -> GameRecord:
    """`model` is what Illusion believes about its opponent, `opponent` is who it really
    plays (defaults to the same model, but they can differ)."""
    human = opponent or model
    board = chess.Board()
    traps: list[tuple[int, str, int]] = []

    while not board.is_game_over(claim_draw=True) and len(board.move_stack) < MAX_PLIES:
        if (board.turn == chess.WHITE) == illusion_is_white:
            choice = swindle_choose(board, engine, model, depth=depth)
            if choice.trap_gain >= TRAP_REPORT_THRESHOLD:
                traps.append((len(board.move_stack) + 1, board.san(choice.move), round(choice.trap_gain)))
            move = choice.move
        else:
            move = human.sample(board, rng)
        board.push(move)
    return GameRecord(board, illusion_is_white, traps)


def find_brilliant_moves(record: GameRecord, evaluator):
    """Replay the game and yield (ply, san, report) for each of Illusion's brilliant moves."""
    replay = chess.Board()
    for ply, move in enumerate(record.board.move_stack, start=1):
        illusion_to_move = (replay.turn == chess.WHITE) == record.illusion_is_white
        if illusion_to_move:
            report = classify(replay, move, evaluator)
            if report.is_brilliant:
                yield ply, replay.san(move), report
        replay.push(move)


def to_pgn(record: GameRecord, opponent_elo: int) -> chess.pgn.Game:
    game = chess.pgn.Game.from_board(record.board)
    human_name = f"SimHuman{opponent_elo}"
    game.headers["White"] = "Illusion" if record.illusion_is_white else human_name
    game.headers["Black"] = human_name if record.illusion_is_white else "Illusion"
    game.headers["Result"] = record.result
    return game


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--games", type=int, default=2)
    parser.add_argument("--opp-elo", type=int, default=1200, help="rating of the simulated opponent")
    parser.add_argument("--my-elo", type=int, default=2000, help="rating maia3 assumes for Illusion")
    parser.add_argument("--depth", type=int, default=3, help="own engine depth (ignored by stockfish)")
    parser.add_argument("--stockfish", metavar="PATH", help="use stockfish as the judge, e.g. 'stockfish'")
    parser.add_argument("--stockfish-depth", type=int, default=12)
    parser.add_argument("--maia3", metavar="MODEL", help="use maia3 as the human model: 5m, 23m or 79m")
    parser.add_argument("--device", default="cpu", help="maia3 device: cpu or cuda")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--annotate", action="store_true", help="report brilliant moves")
    parser.add_argument("--pgn", default="games.pgn")
    return parser.parse_args(argv)


def build_judge(args: argparse.Namespace):
    if args.stockfish:
        return StockfishEngine.from_path(args.stockfish, depth=args.stockfish_depth)
    return Engine(args.depth)


def build_model(args: argparse.Namespace):
    if args.maia3:
        backend = Maia3Backend(args.maia3, args.device)
        return Maia3Model(backend, elo=args.opp_elo, oppo_elo=args.my_elo)
    return EloModel(args.opp_elo)


def build_evaluator(args: argparse.Namespace, judge):
    return judge if args.stockfish else EngineEvaluator(depth=max(2, args.depth - 1))


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    judge, model = build_judge(args), build_model(args)
    evaluator = build_evaluator(args, judge)
    total = 0.0
    with open(args.pgn, "w") as pgn_file:
        for index in range(args.games):
            record = play_game(judge, model, args.depth, index % 2 == 0, rng)
            total += record.illusion_score
            side = "white" if record.illusion_is_white else "black"
            print(f"game {index + 1}: {record.result} (illusion as {side}), traps set: {record.traps}")
            if args.annotate:
                for ply, san, report in find_brilliant_moves(record, evaluator):
                    print(f"   brilliant at ply {ply}: {san}!! "
                          f"(sacrificed {report.sacrifice_cp}cp, {report.loss_vs_best}cp from best)")
            print(to_pgn(record, args.opp_elo), file=pgn_file, end="\n\n")
    print(f"illusion score: {total}/{args.games}")
    if args.stockfish:
        judge.close()


if __name__ == "__main__":
    main()
