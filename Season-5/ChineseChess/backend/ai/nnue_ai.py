"""NNUE AI: Neural network evaluation + C++ alpha-beta search.

Uses the NNUE v2 engine from ChessRL (Candidate 5v2).
Requires the engine_c module to be built (pip install -e ChessRL/engine_c/).
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

from ai.base import BaseAI
from engine.game import Game
from engine.move import Move
from engine.board import RED

# Add ChessRL to path so we can import engine_c
_CHESSRL_DIR = Path(__file__).resolve().parents[2] / ".." / "ChessRL"
if str(_CHESSRL_DIR) not in sys.path:
    sys.path.insert(0, str(_CHESSRL_DIR))

from engine_c._xiangqi import NNUESearchV2, Board as CBoard

# Default weights path
_DEFAULT_WEIGHTS = (
    _CHESSRL_DIR / "training" / "candidate5_v2" / "checkpoints" / "nnue_v2_weights.bin"
)


def _python_board_to_c(game: Game) -> CBoard:
    """Convert Python Board (game.board.grid) to C++ Board."""
    return CBoard(game.board.grid)


class NNUEAI(BaseAI):
    def __init__(self, depth: int = 4, weights_path: str | None = None):
        self.depth = depth
        self._engine = NNUESearchV2()
        path = weights_path or str(_DEFAULT_WEIGHTS)
        if not os.path.exists(path):
            raise FileNotFoundError(f"NNUE weights not found: {path}")
        self._engine.load_weights(path)
        self._engine.set_nnue_weight(1.0)

    @property
    def name(self) -> str:
        return f"NNUE(depth={self.depth})"

    def choose_move(self, game: Game) -> Move:
        c_board = _python_board_to_c(game)
        stm = 1 if game.current_turn == RED else -1

        result = self._engine.search(c_board, stm, self.depth)

        fr, fc = result["from_row"], result["from_col"]
        tr, tc = result["to_row"], result["to_col"]

        # Find the captured piece (if any) for the Move dataclass
        captured = game.board.get(tr, tc)

        return Move(fr, fc, tr, tc, captured=captured)
