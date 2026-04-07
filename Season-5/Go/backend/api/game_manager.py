"""Game session management."""

from __future__ import annotations

import uuid
from typing import Optional

from engine.game import Game
from ai.base import BaseAI
from ai.random_ai import RandomAI
from ai.greedy_ai import GreedyAI
from ai.mcts_ai import MCTSAI


AI_TYPES = {
    "random": lambda sims: RandomAI(),
    "greedy": lambda sims: GreedyAI(),
    "mcts": lambda sims: MCTSAI(simulations=sims),
}


class GameSession:
    def __init__(self, game_id: str, game: Game, player_color: str,
                 ai: Optional[BaseAI], ai_color: Optional[int]):
        self.game_id = game_id
        self.game = game
        self.player_color = player_color
        self.ai = ai
        self.ai_color = ai_color

    def to_state_dict(self) -> dict:
        state = self.game.to_dict()
        state["game_id"] = self.game_id
        state["player_color"] = self.player_color
        state["ai_type"] = self.ai.name if self.ai else "human"
        return state


class GameManager:
    def __init__(self):
        self.games: dict[str, GameSession] = {}

    def create_game(self, player_color: str = "black", ai_type: str = "random",
                    board_size: int = 19, mcts_sims: int = 500) -> GameSession:
        from engine.board import Stone

        game_id = str(uuid.uuid4())
        komi = 7.5 if board_size == 19 else 5.5 if board_size == 9 else 6.5
        game = Game(size=board_size, komi=komi)

        if ai_type == "human":
            ai = None
            ai_color_val = None
        else:
            ai_factory = AI_TYPES.get(ai_type, AI_TYPES["random"])
            ai = ai_factory(mcts_sims)
            ai_color_val = Stone.WHITE if player_color == "black" else Stone.BLACK

        session = GameSession(
            game_id=game_id,
            game=game,
            player_color=player_color,
            ai=ai,
            ai_color=ai_color_val,
        )
        self.games[game_id] = session
        return session

    def get_game(self, game_id: str) -> Optional[GameSession]:
        return self.games.get(game_id)

    def delete_game(self, game_id: str):
        self.games.pop(game_id, None)


game_manager = GameManager()
