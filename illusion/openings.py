"""Opening finder: grows an opening repertoire that scores well against humans.

The idea is the Alien Gambit one: a line can be "bad" for Stockfish and still win
most games, because the human on the other side has to find only-moves. So lines
are judged mostly by their expected score against a human model (maia3 or
`EloModel`), with the engine's own verdict mixed in so the result stays playable.

The search is an expectimax tree:
  * our turn: try a few candidate moves (the engine's best ones plus one
    surprise, usually a sacrifice or a check) and keep the one that scores best
  * their turn: expand the replies a human of the given elo actually plays, until
    `coverage` of their probability is covered; the rest are assumed to be best play
  * leaves: the engine's evaluation, turned into a win chance
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Protocol

import chess
import chess.pgn

from .material import see
from .opponent import MoveProbabilities

SCORE_CLIP = 2000  # centipawns; stops mate scores from wrecking win chances
WIN_CHANCE_SCALE = 0.00368208  # lichess' centipawn -> win chance curve
PASS_THROUGH = "~"  # book.txt prefix for a move the engine walks through but never plays from the book


class Judge(Protocol):
    """Anything that scores positions: `illusion.Engine` or `StockfishEngine`."""

    def value(self, board: chess.Board, depth: int) -> int: ...

    def move_value(self, board: chess.Board, move: chess.Move, depth: int) -> int: ...


class HumanModel(Protocol):
    def distribution(self, board: chess.Board) -> MoveProbabilities: ...


@dataclass
class FinderConfig:
    side: chess.Color = chess.WHITE
    my_moves: int = 4  # how many of our own moves the tree goes deep
    candidates: int = 3  # our moves tried in each position (the last slot is the surprise move)
    coverage: float = 0.75  # expand human replies until this much of their probability is covered
    max_replies: int = 3  # but never more than this many replies per position
    soundness: int = 250  # never play a move this many centipawns worse than the engine's best
    truth_weight: float = 0.25  # 0 = only results vs humans count, 1 = only the engine's eval counts
    novelty_weight: float = 0.05  # bonus for moves humans at your elo rarely play (your own lines)
    trap_loss: int = 150  # a human reply that loses this much is a trap
    trap_probability: float = 0.10  # ...if humans play it at least this often
    judge_depth: int = 3  # only used by the built-in python engine; stockfish has its own depth
    min_line: float = 0.0  # drop opponent lines humans reach less often than this (0 = keep all); keeps the repertoire short


def win_chance(cp: float) -> float:
    """Centipawns (from our side) to a win chance between 0 and 1."""
    cp = max(-SCORE_CLIP, min(SCORE_CLIP, cp))
    return 1 / (1 + math.exp(-WIN_CHANCE_SCALE * cp))


@dataclass
class Reply:
    move: chess.Move
    probability: float  # how often a human of the given elo plays it
    loss: int  # centipawns it throws away versus their best reply
    child: "Node"


@dataclass
class Node:
    """One position in the repertoire. `value` is our expected score (0..1)."""

    ours: bool
    value: float
    engine_cp: int = 0  # engine verdict from our side
    move: chess.Move | None = None  # our move, when it is our turn
    child: "Node | None" = None  # the position after our move
    replies: list[Reply] = field(default_factory=list)  # their likely replies, most likely first
    best_reply: chess.Move | None = None  # what the engine says they should play
    unexpanded: float = 0.0  # probability of replies we did not look at

    def traps(self, config: FinderConfig) -> list[Reply]:
        return [r for r in self.replies if r.probability >= config.trap_probability and r.loss >= config.trap_loss]


class OpeningFinder:
    def __init__(
        self,
        judge: Judge,
        opponent: HumanModel,
        config: FinderConfig | None = None,
        me: HumanModel | None = None,
        progress: Callable[[int], None] | None = None,
    ):
        """`opponent` predicts the human you face; `me` (optional) predicts what humans
        at your own elo play, so rare moves can earn the novelty bonus."""
        self.judge = judge
        self.opponent = opponent
        self.me = me
        self.config = config or FinderConfig()
        self.progress = progress
        self.positions = 0

    # ---- public API -------------------------------------------------------
    def find(self, board: chess.Board | None = None) -> Node:
        """Build the repertoire tree from `board` (the start position by default)."""
        board = (board or chess.Board()).copy()
        self.positions = 0
        if board.turn == self.config.side:
            return self._ours(board, self.config.my_moves, 1.0)
        return self._theirs(board, self.config.my_moves, 1.0)

    # ---- helpers ----------------------------------------------------------
    def _our_cp(self, board: chess.Board) -> int:
        """Engine score of the position from our side."""
        score = self.judge.value(board, self.config.judge_depth)
        return score if board.turn == self.config.side else -score

    def _leaf(self, board: chess.Board) -> Node:
        cp = self._our_cp(board)
        return Node(ours=board.turn == self.config.side, value=win_chance(cp), engine_cp=cp)

    def _tick(self) -> None:
        self.positions += 1
        if self.progress:
            self.progress(self.positions)

    def _ranked_moves(self, board: chess.Board) -> list[tuple[chess.Move, int]]:
        """Every legal move with its engine score (mover's view), best first."""
        ranker = getattr(self.judge, "top_moves", None)
        if ranker is not None:
            return ranker(board)
        scored = [(m, self.judge.move_value(board, m, self.config.judge_depth)) for m in board.legal_moves]
        scored.sort(key=lambda pair: -pair[1])
        return scored

    def _candidates(self, board: chess.Board, ranked: list[tuple[chess.Move, int]]) -> list[tuple[chess.Move, int]]:
        """The engine's best moves plus one surprise: the boldest sacrifice or check still within `soundness`."""
        best = ranked[0][1]
        sound = [(m, cp) for m, cp in ranked if cp >= best - self.config.soundness]
        picks = sound[: max(1, self.config.candidates - 1)]
        rest = [(m, cp) for m, cp in sound if (m, cp) not in picks]
        if len(picks) < self.config.candidates and rest:
            picks.append(min(rest, key=lambda pair: self._boldness(board, pair[0])))
        return picks

    @staticmethod
    def _boldness(board: chess.Board, move: chess.Move) -> int:
        """Lower = bolder. Material given away counts most, then checks."""
        return see(board, move) - (50 if board.gives_check(move) else 0)

    def _human_probability(self, board: chess.Board, move: chess.Move) -> float:
        if self.me is None:
            return 0.0
        return dict(self.me.distribution(board)).get(move, 0.0)

    # ---- the tree ---------------------------------------------------------
    def _ours(self, board: chess.Board, moves_left: int, reach: float) -> Node:
        if moves_left == 0 or board.is_game_over(claim_draw=True):
            return self._leaf(board)
        self._tick()
        config = self.config
        ranked = self._ranked_moves(board)
        best: Node | None = None
        best_pick = -math.inf
        for move, cp in self._candidates(board, ranked):
            novelty = 1.0 - self._human_probability(board, move)
            board.push(move)
            child = self._theirs(board, moves_left, reach)
            board.pop()
            value = (1 - config.truth_weight) * child.value + config.truth_weight * win_chance(cp)
            pick = value + config.novelty_weight * novelty
            if pick > best_pick:
                best_pick = pick
                best = Node(ours=True, value=value, engine_cp=cp, move=move, child=child)
        assert best is not None  # the game is not over, so there is a legal move
        return best

    def _theirs(self, board: chess.Board, moves_left: int, reach: float) -> Node:
        if board.is_game_over(claim_draw=True):
            return self._leaf(board)
        self._tick()
        config = self.config
        ranked = self._ranked_moves(board)
        their_best_move, their_best = ranked[0]
        scores = dict(ranked)
        best_play = win_chance(-their_best)

        node = Node(ours=False, value=0.0, engine_cp=-their_best, best_reply=their_best_move)
        covered = 0.0
        for move, probability in self.opponent.distribution(board):
            if covered >= config.coverage or len(node.replies) >= config.max_replies:
                break
            if reach * probability < config.min_line:
                break  # replies are sorted, so every later one is rarer still
            if move not in scores:  # the ranker skipped it, so score it on its own
                scores[move] = self.judge.move_value(board, move, config.judge_depth)
            board.push(move)
            child = self._ours(board, moves_left - 1, reach * probability)
            board.pop()
            node.replies.append(Reply(move, probability, max(0, their_best - scores[move]), child))
            node.value += probability * child.value
            covered += probability
        node.unexpanded = max(0.0, 1.0 - covered)
        node.value += node.unexpanded * best_play  # replies we skipped: assume best defence
        return node


# ---- reading the tree ------------------------------------------------------
def lines(node: Node, board: chess.Board) -> list[list[tuple[chess.Move, bool]]]:
    """Every path through the tree as (move, is_ours) pairs."""
    if node.ours and node.move is not None and node.child is not None:
        board.push(node.move)
        tails = lines(node.child, board)
        board.pop()
        return [[(node.move, True)] + tail for tail in tails]
    if not node.ours and node.replies:
        result = []
        for reply in node.replies:
            board.push(reply.move)
            result += [[(reply.move, False)] + tail for tail in lines(reply.child, board)]
            board.pop()
        return result
    return [[]]


def positions_to_learn(node: Node) -> int:
    """How many of our moves you need to memorise to play the whole tree."""
    if node.ours:
        return (1 + positions_to_learn(node.child)) if node.child else 0
    return sum(positions_to_learn(r.child) for r in node.replies)


def collect_traps(node: Node, board: chess.Board, config: FinderConfig):
    """Yield (board before the trap, trap reply, engine's best reply) through the whole tree."""
    if node.ours:
        if node.move is not None and node.child is not None:
            board.push(node.move)
            yield from collect_traps(node.child, board, config)
            board.pop()
        return
    trap_moves = {id(r) for r in node.traps(config)}
    for reply in node.replies:
        if id(reply) in trap_moves:
            yield board.copy(), reply, node.best_reply
        board.push(reply.move)
        yield from collect_traps(reply.child, board, config)
        board.pop()


def book_text(tree: Node, start: chess.Board, config: FinderConfig, title: str) -> str:
    """Lines for the engine's book file. Their moves get the `~` prefix so the engine
    walks through them but never plays a human's blunder from the book itself."""
    side = "white" if config.side == chess.WHITE else "black"
    out = [
        f"# {title}: generated by openings.py, Illusion plays {side}",
        f"# {PASS_THROUGH}move = opponent move, walked through but never played from the book",
    ]
    prefix_board = chess.Board()
    prefix = []
    for move in start.move_stack:
        ours = prefix_board.turn == config.side
        prefix.append(("" if ours else PASS_THROUGH) + prefix_board.san(move))
        prefix_board.push(move)
    for path in lines(tree, start.copy()):
        board = start.copy()
        tokens = list(prefix)
        for move, ours in path:
            tokens.append(("" if ours else PASS_THROUGH) + board.san(move))
            board.push(move)
        if tokens:
            out.append(" ".join(tokens))
    return "\n".join(out) + "\n"


def to_pgn(tree: Node, start: chess.Board, config: FinderConfig, title: str, opp_elo: int) -> chess.pgn.Game:
    """The repertoire as one PGN with variations, trap replies commented."""
    side = "White" if config.side == chess.WHITE else "Black"
    game = chess.pgn.Game()
    game.headers["Event"] = title
    game.headers["White" if config.side == chess.WHITE else "Black"] = "You"
    game.headers["Black" if config.side == chess.WHITE else "White"] = f"Human {opp_elo}"
    game.headers["Opening"] = title
    game.comment = (
        f"{title} for {side}. Expected score vs {opp_elo}s: {tree.value:.0%}. "
        f"Moves to memorise: {positions_to_learn(tree)}."
    )
    node: chess.pgn.GameNode = game
    for move in start.move_stack:
        node = node.add_main_variation(move)
    _add_pgn(node, tree, config, opp_elo)
    return game


def _add_pgn(pgn: chess.pgn.GameNode, node: Node, config: FinderConfig, opp_elo: int) -> None:
    if node.ours:
        if node.move is None or node.child is None:
            return
        child = pgn.add_variation(node.move)
        child.comment = f"scores {node.value:.0%} vs humans, engine {node.engine_cp / 100:+.1f}"
        _add_pgn(child, node.child, config, opp_elo)
        return
    trap_ids = {id(r) for r in node.traps(config)}
    for reply in node.replies:
        child = pgn.add_variation(reply.move)
        notes = [f"{opp_elo}s play this {reply.probability:.0%}"]
        if id(reply) in trap_ids:
            child.nags.add(chess.pgn.NAG_MISTAKE if reply.loss < 300 else chess.pgn.NAG_BLUNDER)
            best = pgn.board().san(node.best_reply) if node.best_reply else "?"
            notes.append(f"TRAP, loses {reply.loss / 100:.1f}, only good answer was {best}")
        child.comment = ", ".join(notes)
        _add_pgn(child, reply.child, config, opp_elo)


# ---- readable output --------------------------------------------------------
@dataclass
class Step:
    move: chess.Move
    ours: bool
    probability: float = 1.0  # how often humans play it (their moves only)
    trap: bool = False
    loss: int = 0  # centipawns their move throws away
    best: str = ""  # the only good answer, in SAN


def walk_lines(node: Node, board: chess.Board, config: FinderConfig, steps: tuple = (), reach: float = 1.0):
    """Yield (steps, reach) for every line in the tree. `reach` is how often humans get into it."""
    if node.ours:
        if node.move is None or node.child is None:
            yield list(steps), reach
            return
        board.push(node.move)
        yield from walk_lines(node.child, board, config, steps + (Step(node.move, True),), reach)
        board.pop()
        return
    if not node.replies:
        yield list(steps), reach
        return
    trap_ids = {id(r) for r in node.traps(config)}
    best = board.san(node.best_reply) if node.best_reply else ""
    for reply in node.replies:
        step = Step(reply.move, False, reply.probability, id(reply) in trap_ids, reply.loss, best)
        board.push(reply.move)
        yield from walk_lines(reply.child, board, config, steps + (step,), reach * reply.probability)
        board.pop()


def _numbered(start: chess.Board, steps: list[Step], marks: bool = True) -> tuple[str, chess.Board]:
    """'1.Nf3 d5 2.c4 dxc4?' style text for a line (the start line included) and the board it ends on."""
    board = chess.Board()
    tokens: list[str] = []
    played = list(start.move_stack) + [step.move for step in steps]
    flags = [False] * len(start.move_stack) + [step.trap and marks for step in steps]
    for move, flagged in zip(played, flags):
        san = board.san(move)
        if board.turn == chess.WHITE:
            tokens.append(f"{board.fullmove_number}.{san}")
        else:
            tokens.append(f"{board.fullmove_number}...{san}" if not tokens else san)
        if flagged:
            tokens[-1] += "?"
        board.push(move)
    return " ".join(tokens), board


def cheat_sheet(tree: Node, start: chess.Board, config: FinderConfig, title: str, opp_elo: int) -> str:
    """Plain text you can read on a phone: one line per row, most common first, traps marked."""
    rows = sorted(walk_lines(tree, start.copy(), config), key=lambda row: -row[1])
    side = "White" if config.side == chess.WHITE else "Black"
    out = [
        title.upper(),
        f"you play {side}. expected score vs {opp_elo}s: {tree.value:.0%}",
        f"{len(rows)} lines, {positions_to_learn(tree)} of your moves to learn",
        "",
        "LINES (most common first, % = how often humans end up here)",
        "",
    ]
    traps: dict[tuple[str, chess.Move], Step] = {}
    trap_text: dict[tuple[str, chess.Move], str] = {}
    for number, (steps, reach) in enumerate(rows, start=1):
        out.append(f"{number}. [{reach:.0%}] {_numbered(start, steps)[0]}")
        for index, step in enumerate(steps):
            if step.trap:
                before, board = _numbered(start, steps[:index], marks=False)
                key = (before, step.move)
                traps[key] = step
                trap_text[key] = board.san(step.move)
    if traps:
        out += ["", "TRAPS (the reply humans fall for)", ""]
        for key, step in sorted(traps.items(), key=lambda item: -item[1].probability * item[1].loss):
            out.append(f"after {key[0] or '(start)'}:")
            out.append(f"  humans play {trap_text[key]}? {step.probability:.0%} of the time, it loses {step.loss / 100:.1f}")
            out.append(f"  the only good move is {step.best}")
            out.append("")
    return "\n".join(out).rstrip() + "\n"


def study_pgn(tree: Node, start: chess.Board, config: FinderConfig, title: str, opp_elo: int) -> str:
    """One short game per line (a lichess study chapter each), no nested brackets."""
    rows = sorted(walk_lines(tree, start.copy(), config), key=lambda row: -row[1])
    games = []
    for number, (steps, reach) in enumerate(rows, start=1):
        game = chess.pgn.Game()
        game.headers["Event"] = f"{title} {number}"
        game.headers["Round"] = str(number)
        game.headers["White"] = "You" if config.side == chess.WHITE else f"Human {opp_elo}"
        game.headers["Black"] = "You" if config.side == chess.BLACK else f"Human {opp_elo}"
        game.comment = f"humans reach this line {reach:.0%} of the time"
        node: chess.pgn.GameNode = game
        for move in start.move_stack:
            node = node.add_main_variation(move)
        for step in steps:
            node = node.add_main_variation(step.move)
            if step.trap:
                node.nags.add(chess.pgn.NAG_MISTAKE if step.loss < 300 else chess.pgn.NAG_BLUNDER)
                node.comment = f"TRAP: {step.probability:.0%} play this, it loses {step.loss / 100:.1f}. only good move: {step.best}"
        games.append(str(game))
    return "\n\n".join(games) + "\n"
