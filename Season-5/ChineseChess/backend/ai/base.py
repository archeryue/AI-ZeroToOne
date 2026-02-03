"""Abstract AI interface for Chinese Chess."""

from abc import ABC, abstractmethod
from engine.game import Game
from engine.move import Move


class BaseAI(ABC):
    """Abstract base class for AI opponents."""

    @abstractmethod
    def choose_move(self, game: Game) -> Move:
        """Choose a move given the current game state.

        The game's current_turn indicates which color the AI plays.
        Must return a legal move.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this AI."""
        ...
