"""Play Illusion against a simulated human of a given elo.

    python arena.py --games 2 --opp-elo 1200 --depth 3 --annotate
"""
from __future__ import annotations

import argparse
import random
from dataclasses import dataclass

import chess
import chess.pgn

from illusion import EloModel, Engine, EngineEvaluator, classify, swindle_choose

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


def play_game(depth: int, opponent_elo: int, illusion_is_white: bool, rng: random.Random) -> GameRecord:
    board = chess.Board()
    engine = Engine(depth)
    human = EloModel(opponent_elo)
    traps: list[tuple[int, str, int]] = []

    while not board.is_game_over(claim_draw=True) and len(board.move_stack) < MAX_PLIES:
        if (board.turn == chess.WHITE) == illusion_is_white:
            choice = swindle_choose(board, engine, human, depth=depth)
            if choice.trap_gain >= TRAP_REPORT_THRESHOLD:
                traps.append((len(board.move_stack) + 1, board.san(choice.move), round(choice.trap_gain)))
            move = choice.move
        else:
            move = human.sample(board, rng)
        board.push(move)
    return GameRecord(board, illusion_is_white, traps)


def find_brilliant_moves(record: GameRecord, depth: int):
    """Replay the game and yield (ply, san, report) for each of Illusion's brilliant moves."""
    evaluator = EngineEvaluator(depth=max(2, depth - 1))
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=2)
    parser.add_argument("--opp-elo", type=int, default=1200)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--annotate", action="store_true", help="report brilliant moves")
    parser.add_argument("--pgn", default="games.pgn")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    total = 0.0
    with open(args.pgn, "w") as pgn_file:
        for index in range(args.games):
            record = play_game(args.depth, args.opp_elo, index % 2 == 0, rng)
            total += record.illusion_score
            side = "white" if record.illusion_is_white else "black"
            print(f"game {index + 1}: {record.result} (illusion as {side}), traps set: {record.traps}")
            if args.annotate:
                for ply, san, report in find_brilliant_moves(record, args.depth):
                    print(f"   brilliant at ply {ply}: {san}!! "
                          f"(sacrificed {report.sacrifice_cp}cp, {report.loss_vs_best}cp from best)")
            print(to_pgn(record, args.opp_elo), file=pgn_file, end="\n\n")
    print(f"illusion score: {total}/{args.games}")


if __name__ == "__main__":
    main()
