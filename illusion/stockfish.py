"""Stockfish as a judge: scores positions for the swindle chooser and the brilliant detector."""
from __future__ import annotations

import chess
import chess.engine

from .search import MATE

DEFAULT_DEPTH = 12
MAX_CACHED_POSITIONS = 50_000
DEFAULT_TOP_MOVES = 12


class StockfishEngine:
    """Same interface as `illusion.Engine` (`value` / `move_value`), so it can be
    dropped into `swindle_choose` and `EngineEvaluator`-style checks.

    Stockfish always searches to its own fixed `depth`; the `depth` argument that
    callers pass is accepted for compatibility and ignored.
    """

    def __init__(self, uci_engine: chess.engine.SimpleEngine, depth: int = DEFAULT_DEPTH):
        self._engine = uci_engine
        self._limit = chess.engine.Limit(depth=depth)
        self._cache: dict[str, int] = {}
        self._ranked: dict[str, list[tuple[chess.Move, int]]] = {}

    @classmethod
    def from_path(
        cls, path: str = "stockfish", depth: int = DEFAULT_DEPTH, threads: int = 1, hash_mb: int = 64
    ) -> "StockfishEngine":
        uci_engine = chess.engine.SimpleEngine.popen_uci(path)
        try:
            wanted = {"Threads": threads, "Hash": hash_mb}
            uci_engine.configure({name: value for name, value in wanted.items() if name in uci_engine.options})
        except Exception:
            uci_engine.quit()  # never leave the engine process running if setup fails
            raise
        return cls(uci_engine, depth)

    def value(self, board: chess.Board, depth: int | None = None) -> int:
        """Centipawns from the side to move's view (mates are +/- MATE minus distance)."""
        if board.is_checkmate():
            return -MATE
        if board.is_game_over(claim_draw=True):
            return 0
        key = board.fen()
        if key not in self._cache:
            if len(self._cache) >= MAX_CACHED_POSITIONS:
                self._cache.clear()
            info = self._engine.analyse(board, self._limit)
            self._cache[key] = info["score"].pov(board.turn).score(mate_score=MATE)
        return self._cache[key]

    def top_moves(self, board: chess.Board, count: int = DEFAULT_TOP_MOVES) -> list[tuple[chess.Move, int]]:
        """The engine's `count` best moves with their scores (mover's view), best first.
        One multipv search instead of one search per move."""
        key = f"{board.fen()}|{count}"
        if key not in self._ranked:
            if len(self._ranked) >= MAX_CACHED_POSITIONS:
                self._ranked.clear()
            infos = self._engine.analyse(board, self._limit, multipv=count)
            ranked = [(info["pv"][0], info["score"].pov(board.turn).score(mate_score=MATE)) for info in infos if info.get("pv")]
            ranked.sort(key=lambda pair: -pair[1])
            self._ranked[key] = ranked
        return list(self._ranked[key])

    def move_value(self, board: chess.Board, move: chess.Move, depth: int | None = None) -> int:
        """Score of playing `move`, from the mover's view."""
        after = board.copy()
        after.push(move)
        return -self.value(after)

    def close(self) -> None:
        self._engine.quit()

    def __enter__(self) -> "StockfishEngine":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
