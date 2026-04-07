"""Base AI interface for Go."""

from abc import ABC, abstractmethod
from engine.game import Game


class BaseAI(ABC):
    @abstractmethod
    def choose_move(self, game: Game) -> tuple[int, int]:
        """Choose a move. Returns (row, col) or (-1, -1) for pass."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...
