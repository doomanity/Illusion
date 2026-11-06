"""Stockfish as a judge: scores positions for the swindle chooser and the brilliant detector."""
from __future__ import annotations

import chess
import chess.engine

from .search import MATE

DEFAULT_DEPTH = 12
MAX_CACHED_POSITIONS = 50_000


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

    @classmethod
    def from_path(
        cls, path: str = "stockfish", depth: int = DEFAULT_DEPTH, threads: int = 1, hash_mb: int = 64
    ) -> "StockfishEngine":
        uci_engine = chess.engine.SimpleEngine.popen_uci(path)
        uci_engine.configure({"Threads": threads, "Hash": hash_mb})
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
