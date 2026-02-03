"""AI that picks a random legal move."""

import random
from ai.base import BaseAI
from engine.game import Game
from engine.move import Move


class RandomAI(BaseAI):
    @property
    def name(self) -> str:
        return "Random"

    def choose_move(self, game: Game) -> Move:
        legal = game.get_legal_moves()
        return random.choice(legal)
