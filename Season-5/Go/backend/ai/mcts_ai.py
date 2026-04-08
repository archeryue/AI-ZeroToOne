"""Pure MCTS AI: Monte Carlo Tree Search with random rollouts (no neural network).

This is a classical MCTS implementation using UCT (Upper Confidence bounds for Trees).
Even without a neural network, MCTS with random rollouts plays reasonable Go.
"""

import math
import random
import time
from typing import Optional

from engine.board import Stone, PASS, RESIGN
from engine.game import Game, GameStatus
from .base import BaseAI


class MCTSNode:
    __slots__ = ['parent', 'move', 'children', 'visits', 'wins', 'untried_moves']

    def __init__(self, parent: Optional['MCTSNode'], move: tuple[int, int],
                 untried_moves: list[tuple[int, int]]):
        self.parent = parent
        self.move = move
        self.children: list[MCTSNode] = []
        self.visits = 0
        self.wins = 0.0
        self.untried_moves = untried_moves

    def uct_score(self, explore: float = 1.414) -> float:
        if self.visits == 0:
            return float('inf')
        return (self.wins / self.visits) + explore * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )

    def best_child(self, explore: float = 1.414) -> 'MCTSNode':
        return max(self.children, key=lambda c: c.uct_score(explore))

    def most_visited_child(self) -> 'MCTSNode':
        return max(self.children, key=lambda c: c.visits)


class MCTSAI(BaseAI):
    # MCTS uses its own win rate — resign when below this threshold
    RESIGN_WIN_RATE_THRESHOLD = 0.10

    def __init__(self, simulations: int = 500, max_time: float = 5.0):
        self.simulations = simulations
        self.max_time = max_time

    def choose_move(self, game: Game) -> tuple[int, int]:
        legal = game.get_legal_moves()
        if not legal:
            return PASS

        # For very early game or very few options, skip heavy search
        if len(legal) == 1:
            return legal[0]

        root_color = game.current_turn
        # Add pass as a possible move
        all_moves = legal + [PASS]

        root = MCTSNode(parent=None, move=PASS, untried_moves=all_moves[:])

        start_time = time.time()
        # Track overall win rate for resign evaluation
        total_rollouts = 0
        total_wins = 0.0

        for i in range(self.simulations):
            if time.time() - start_time > self.max_time:
                break

            # Clone game state
            sim_game = self._clone_game(game)

            # Selection
            node = root
            while not node.untried_moves and node.children:
                node = node.best_child()
                if node.move == PASS:
                    sim_game.make_move(-1, -1)
                else:
                    try:
                        sim_game.make_move(node.move[0], node.move[1])
                    except ValueError:
                        break

            # Expansion
            if node.untried_moves and sim_game.status == GameStatus.PLAYING:
                move = random.choice(node.untried_moves)
                node.untried_moves.remove(move)

                if move == PASS:
                    sim_game.make_move(-1, -1)
                else:
                    try:
                        sim_game.make_move(move[0], move[1])
                    except ValueError:
                        # Illegal move (can happen due to ko changes), skip
                        continue

                child_legal = sim_game.get_legal_moves() + [PASS]
                child = MCTSNode(parent=node, move=move, untried_moves=child_legal)
                node.children.append(child)
                node = child

            # Simulation (random rollout)
            result = self._rollout(sim_game, root_color)
            total_rollouts += 1
            total_wins += result

            # Backpropagation
            while node is not None:
                node.visits += 1
                node.wins += result
                # Flip result for opponent's nodes
                result = 1.0 - result
                node = node.parent

        if not root.children:
            return random.choice(legal) if legal else PASS

        best = root.most_visited_child()

        # --- Resign check (uses MCTS win rate — essentially free) ---
        if total_rollouts > 0 and self._should_invoke_resign_check(game):
            win_rate = total_wins / total_rollouts
            if win_rate < self.RESIGN_WIN_RATE_THRESHOLD:
                return RESIGN

        return best.move

    def _should_invoke_resign_check(self, game: Game) -> bool:
        """MCTS can check every move since win rate is already computed."""
        move_count = len(game.move_history)
        return move_count >= max(20, game.size * 2)

    def _rollout(self, game: Game, root_color: int, max_moves: int = 200) -> float:
        """Random playout until game ends or max_moves reached."""
        moves = 0
        consecutive_passes = game.consecutive_passes

        while game.status == GameStatus.PLAYING and moves < max_moves:
            legal = game.get_legal_moves()

            if not legal or random.random() < 0.1:
                # Pass
                game.make_move(-1, -1)
                consecutive_passes += 1
                if consecutive_passes >= 2:
                    break
            else:
                move = random.choice(legal)
                try:
                    game.make_move(move[0], move[1])
                    consecutive_passes = 0
                except ValueError:
                    game.make_move(-1, -1)
                    consecutive_passes += 1

            moves += 1

        # Score
        if game.status == GameStatus.BLACK_WIN:
            return 1.0 if root_color == Stone.BLACK else 0.0
        elif game.status == GameStatus.WHITE_WIN:
            return 1.0 if root_color == Stone.WHITE else 0.0
        else:
            # Game didn't end — score current position
            black_score, white_score = game.board.score(game.komi)
            if root_color == Stone.BLACK:
                return 1.0 if black_score > white_score else 0.0
            else:
                return 1.0 if white_score > black_score else 0.0

    @property
    def name(self) -> str:
        return f"MCTS ({self.simulations} sims)"
