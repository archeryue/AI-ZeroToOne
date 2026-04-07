"""Random AI: picks a random legal move."""

import random
from engine.game import Game
from .base import BaseAI


class RandomAI(BaseAI):
    def choose_move(self, game: Game) -> tuple[int, int]:
        legal = game.get_legal_moves()
        if not legal:
            return (-1, -1)  # pass
        # Small chance to pass even if moves exist (mimics awareness of pass)
        if random.random() < 0.02:
            return (-1, -1)
        return random.choice(legal)

    @property
    def name(self) -> str:
        return "Random"
