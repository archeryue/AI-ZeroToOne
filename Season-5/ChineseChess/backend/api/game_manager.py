"""In-memory game session store."""

from __future__ import annotations

import uuid
from typing import Optional

from engine.game import Game
from ai.base import BaseAI
from ai.random_ai import RandomAI
from ai.greedy_ai import GreedyAI
from ai.minimax_ai import MinimaxAI
from engine.board import RED, BLACK


AI_TYPES = {
    "random": lambda depth: RandomAI(),
    "greedy": lambda depth: GreedyAI(),
    "minimax": lambda depth: MinimaxAI(depth=depth),
}


class GameSession:
    """Wraps a Game with AI and player configuration."""

    def __init__(self, game_id: str, player_color: int, ai: Optional[BaseAI]):
        self.game_id = game_id
        self.game = Game()
        self.player_color = player_color
        self.ai = ai
        self.ai_color = -player_color if ai else None

    def to_state_dict(self) -> dict:
        """Get full game state dict with session metadata."""
        state = self.game.to_dict()
        state["game_id"] = self.game_id
        state["player_color"] = "red" if self.player_color == RED else "black"
        state["ai_type"] = self.ai.name if self.ai else "human"
        return state


class GameManager:
    """Manages all active game sessions."""

    def __init__(self):
        self.games: dict[str, GameSession] = {}

    def create_game(
        self, player_color: str = "red", ai_type: str = "greedy", ai_depth: int = 3
    ) -> GameSession:
        game_id = str(uuid.uuid4())[:8]
        color = RED if player_color == "red" else BLACK

        ai = None
        if ai_type != "human":
            ai_factory = AI_TYPES.get(ai_type, AI_TYPES["greedy"])
            ai = ai_factory(ai_depth)

        session = GameSession(game_id, color, ai)
        self.games[game_id] = session
        return session

    def get_game(self, game_id: str) -> Optional[GameSession]:
        return self.games.get(game_id)

    def delete_game(self, game_id: str) -> bool:
        if game_id in self.games:
            del self.games[game_id]
            return True
        return False


# Singleton instance
game_manager = GameManager()
