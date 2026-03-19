"""Monte Carlo Tree Search for AlphaZero Chinese Chess.

Uses PUCT selection (Polynomial Upper Confidence Trees) guided by the neural
network's policy and value predictions. No rollouts — leaf evaluation is
done entirely by the value head.

Key design:
- Each node stores visit count N, total value W, prior P, and children
- Selection: pick child maximizing Q + U (PUCT formula)
- Expansion: when we reach a leaf, evaluate with NN, create children
- Backup: propagate value up the tree, flipping sign at each level
- The tree persists across simulations within a single move decision
"""

import math
import numpy as np

try:
    import engine_c as cc
    _USE_CPP = True
except ImportError:
    _USE_CPP = False

from engine.board import RED, BLACK
from engine.game import Game, GameStatus
from engine.rules import get_legal_moves
from env.action_space import encode_move, decode_action, get_action_mask, NUM_ACTIONS
from env.observation import board_to_observation

# Map Python constants to C++ constants
if _USE_CPP:
    _STATUS_MAP = {
        cc.STATUS_PLAYING: GameStatus.PLAYING,
        cc.STATUS_RED_WIN: GameStatus.RED_WIN,
        cc.STATUS_BLACK_WIN: GameStatus.BLACK_WIN,
        cc.STATUS_DRAW: GameStatus.DRAW,
    }


class MCTSNode:
    """A node in the MCTS tree."""

    __slots__ = ['parent', 'action', 'prior', 'children',
                 'visit_count', 'total_value']

    def __init__(self, parent=None, action: int = -1, prior: float = 0.0):
        self.parent = parent
        self.action = action       # action that led to this node
        self.prior = prior         # P(s,a) from NN policy
        self.children = []         # list of MCTSNode
        self.visit_count = 0       # N(s,a)
        self.total_value = 0.0     # W(s,a)

    @property
    def q_value(self) -> float:
        """Q(s,a) = W(s,a) / N(s,a)"""
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def select_child(self, c_puct: float) -> 'MCTSNode':
        """Select child with highest PUCT score."""
        sqrt_parent = math.sqrt(self.visit_count)

        best_score = float('-inf')
        best_child = None

        for child in self.children:
            # Q + U where U = c_puct * P * sqrt(N_parent) / (1 + N_child)
            q = child.q_value
            u = c_puct * child.prior * sqrt_parent / (1 + child.visit_count)
            score = q + u

            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    def expand(self, policy: np.ndarray, legal_actions: list[int]):
        """Expand node with children for each legal action."""
        legal_priors = policy[legal_actions]
        prior_sum = legal_priors.sum()
        if prior_sum > 0:
            legal_priors = legal_priors / prior_sum
        else:
            legal_priors = np.ones(len(legal_actions)) / len(legal_actions)

        for i, action in enumerate(legal_actions):
            child = MCTSNode(parent=self, action=action, prior=float(legal_priors[i]))
            self.children.append(child)

    def backup(self, value: float):
        """Propagate value up to root, flipping sign at each level."""
        node = self
        v = value
        while node is not None:
            node.visit_count += 1
            node.total_value += v
            v = -v  # flip for opponent's perspective
            node = node.parent


def _simulate_game(game, action: int):
    """Create a new Game after applying an action (lightweight copy)."""
    if _USE_CPP and isinstance(game, cc.Game):
        return game.simulate_action(action)

    fr, fc, tr, tc = decode_action(action)

    new_game = Game.__new__(Game)
    new_game.board = game.board.copy()
    new_game.current_turn = game.current_turn
    new_game.move_history = []
    new_game.board_history = []
    new_game.status = game.status

    # Apply move directly (skip legality check — we know it's legal)
    piece = new_game.board.get(fr, fc)
    new_game.board.set(fr, fc, 0)
    new_game.board.set(tr, tc, piece)
    new_game.current_turn = -new_game.current_turn

    # Check game over
    new_game._check_game_over()

    return new_game


def _get_legal_actions(game) -> list[int]:
    """Get legal action indices for current player."""
    if _USE_CPP and isinstance(game, cc.Game):
        return cc.get_legal_action_indices(game.board, game.current_turn)

    actions = []
    for move in get_legal_moves(game.board, game.current_turn):
        actions.append(encode_move(move.from_row, move.from_col, move.to_row, move.to_col))
    return actions


def _get_obs(game) -> np.ndarray:
    """Get observation for current game state."""
    if _USE_CPP and isinstance(game, cc.Game):
        return cc.board_to_observation(game.board, game.current_turn)
    return board_to_observation(game.board, game.current_turn)


def _get_mask(game) -> np.ndarray:
    """Get action mask for current game state."""
    if _USE_CPP and isinstance(game, cc.Game):
        return cc.get_action_mask(game.board, game.current_turn)
    return get_action_mask(game.board, game.current_turn)


def _is_playing(game) -> bool:
    """Check if game is still in progress."""
    if _USE_CPP and isinstance(game, cc.Game):
        return game.status == cc.STATUS_PLAYING
    return game.status == GameStatus.PLAYING


def _get_terminal_value(game) -> float:
    """Get value for terminal game state from current player's perspective."""
    if _USE_CPP and isinstance(game, cc.Game):
        if game.status == cc.STATUS_RED_WIN:
            return -1.0 if game.current_turn == cc.BLACK else 1.0
        elif game.status == cc.STATUS_BLACK_WIN:
            return -1.0 if game.current_turn == cc.RED else 1.0
        return 0.0
    return _terminal_value(game)


class MCTS:
    """Monte Carlo Tree Search with neural network guidance."""

    def __init__(self, network, device, num_simulations: int = 50,
                 c_puct: float = 1.5, dirichlet_alpha: float = 0.3,
                 dirichlet_epsilon: float = 0.25):
        self.network = network
        self.device = device
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon

    def search(self, game: Game, add_noise: bool = True) -> np.ndarray:
        """Run MCTS from the current game state.

        Args:
            game: current Game object
            add_noise: if True, add Dirichlet noise at root for exploration

        Returns:
            action_probs: (NUM_ACTIONS,) visit count distribution
        """
        root = MCTSNode()

        # Expand root with NN evaluation
        obs = _get_obs(game)
        mask = _get_mask(game)
        policy, value = self.network.predict(obs, mask, self.device)

        legal_actions = _get_legal_actions(game)
        if not legal_actions:
            return np.zeros(NUM_ACTIONS)

        root.expand(policy, legal_actions)

        # Add Dirichlet noise at root for exploration
        if add_noise and len(root.children) > 0:
            noise = np.random.dirichlet([self.dirichlet_alpha] * len(root.children))
            for i, child in enumerate(root.children):
                child.prior = (1 - self.dirichlet_epsilon) * child.prior + \
                              self.dirichlet_epsilon * noise[i]

        # Cache game states keyed by node id
        game_states = {id(root): game}

        for _ in range(self.num_simulations):
            node = root
            current_game = game

            # Selection: traverse tree using PUCT until we hit a leaf
            while not node.is_leaf():
                node = node.select_child(self.c_puct)
                if id(node) in game_states:
                    current_game = game_states[id(node)]
                else:
                    current_game = _simulate_game(current_game, node.action)
                    game_states[id(node)] = current_game

            # Terminal check
            if not _is_playing(current_game):
                node.backup(_get_terminal_value(current_game))
                continue

            # Expansion: evaluate leaf with NN
            obs = _get_obs(current_game)
            mask = _get_mask(current_game)
            policy, value = self.network.predict(obs, mask, self.device)

            leaf_legal = _get_legal_actions(current_game)
            if leaf_legal:
                node.expand(policy, leaf_legal)

            # Backup value from current player's perspective
            node.backup(value)

        # Extract visit counts
        action_probs = np.zeros(NUM_ACTIONS, dtype=np.float32)
        for child in root.children:
            action_probs[child.action] = child.visit_count

        total = action_probs.sum()
        if total > 0:
            action_probs /= total

        return action_probs

    def select_action(self, game: Game, temperature: float = 1.0,
                      add_noise: bool = True) -> tuple[int, np.ndarray]:
        """Run MCTS and select an action.

        Args:
            game: current Game state
            temperature: controls exploration (1.0 = proportional, ~0 = greedy)
            add_noise: Dirichlet noise at root

        Returns:
            action: selected action index
            action_probs: full visit distribution (training target)
        """
        action_probs = self.search(game, add_noise=add_noise)

        if temperature < 0.01:
            # Greedy
            action = int(np.argmax(action_probs))
        else:
            # Sample proportional to visit_count^(1/temp)
            probs = action_probs ** (1.0 / temperature)
            prob_sum = probs.sum()
            if prob_sum > 0:
                probs /= prob_sum
                action = int(np.random.choice(NUM_ACTIONS, p=probs))
            else:
                legal = _get_legal_actions(game)
                action = int(np.random.choice(legal))

        return action, action_probs


# ---------------------------------------------------------------------------
# Virtual-loss batched MCTS search (for parallel self-play workers)
# ---------------------------------------------------------------------------

def _apply_virtual_loss(node):
    """Apply virtual loss from node up to root to discourage re-selection."""
    n = node
    while n is not None:
        n.visit_count += 1
        n.total_value -= 1.0
        n = n.parent


def _remove_virtual_loss(node):
    """Remove virtual loss from node up to root."""
    n = node
    while n is not None:
        n.visit_count -= 1
        n.total_value += 1.0
        n = n.parent


def _terminal_value(game):
    """Get value for terminal game state from current player's perspective."""
    if game.status == GameStatus.RED_WIN:
        return -1.0 if game.current_turn == BLACK else 1.0
    elif game.status == GameStatus.BLACK_WIN:
        return -1.0 if game.current_turn == RED else 1.0
    return 0.0


def batched_mcts_search(game, evaluate_fn, num_sims=50, virtual_loss_n=8,
                        c_puct=1.5, dirichlet_alpha=0.3, dirichlet_epsilon=0.25,
                        temperature=1.0, add_noise=True):
    """MCTS search with virtual loss for batched NN evaluation.

    Instead of evaluating one leaf at a time, selects multiple leaves per
    batch using virtual loss to diversify paths, then evaluates all leaves
    in one batched call.

    Args:
        game: current Game state
        evaluate_fn: callable(obs_list, mask_list) -> (policies, values)
            obs_list: list of (15,10,9) numpy arrays
            mask_list: list of (8100,) bool numpy arrays
            Returns: policies (N,8100) numpy, values (N,) numpy
        num_sims: total MCTS simulations
        virtual_loss_n: how many leaves to collect per batch
        c_puct: PUCT exploration constant
        temperature: action selection temperature
        add_noise: whether to add Dirichlet noise at root

    Returns:
        action: selected action index
        action_probs: (8100,) visit count distribution (training target)
    """
    root = MCTSNode()
    game_states = {id(root): game}

    # Only one legal move — skip search
    legal_actions = _get_legal_actions(game)
    if not legal_actions:
        return 0, np.zeros(NUM_ACTIONS, dtype=np.float32)
    if len(legal_actions) == 1:
        probs = np.zeros(NUM_ACTIONS, dtype=np.float32)
        probs[legal_actions[0]] = 1.0
        return legal_actions[0], probs

    # Expand root
    obs = _get_obs(game)
    mask = _get_mask(game)
    policies, values = evaluate_fn([obs], [mask])
    root.expand(policies[0], legal_actions)

    # Add Dirichlet noise at root
    if add_noise and root.children:
        noise = np.random.dirichlet([dirichlet_alpha] * len(root.children))
        for i, child in enumerate(root.children):
            child.prior = ((1 - dirichlet_epsilon) * child.prior
                           + dirichlet_epsilon * noise[i])

    # Run simulations in batches of virtual_loss_n
    sims_done = 0
    while sims_done < num_sims:
        batch_size = min(virtual_loss_n, num_sims - sims_done)

        pending_nodes = []
        pending_games = []

        for _ in range(batch_size):
            node = root
            current_game = game

            # Selection: follow PUCT to a leaf
            while not node.is_leaf():
                node = node.select_child(c_puct)
                if id(node) in game_states:
                    current_game = game_states[id(node)]
                else:
                    current_game = _simulate_game(current_game, node.action)
                    game_states[id(node)] = current_game

            # Terminal node — backup immediately, no NN eval needed
            if not _is_playing(current_game):
                node.backup(_get_terminal_value(current_game))
                sims_done += 1
                continue

            # Apply virtual loss so next traversal avoids this path
            _apply_virtual_loss(node)
            pending_nodes.append(node)
            pending_games.append(current_game)
            sims_done += 1

        # Batch evaluate all pending leaves
        if pending_nodes:
            obs_batch = [_get_obs(g) for g in pending_games]
            mask_batch = [_get_mask(g) for g in pending_games]
            policies, values = evaluate_fn(obs_batch, mask_batch)

            for j, (node, current_game) in enumerate(zip(pending_nodes,
                                                          pending_games)):
                # Remove virtual loss
                _remove_virtual_loss(node)

                # Expand
                leaf_legal = _get_legal_actions(current_game)
                if leaf_legal:
                    node.expand(policies[j], leaf_legal)

                # Backup real value
                node.backup(float(values[j]))

    # Extract visit distribution
    action_probs = np.zeros(NUM_ACTIONS, dtype=np.float32)
    for child in root.children:
        action_probs[child.action] = child.visit_count
    total = action_probs.sum()
    if total > 0:
        action_probs /= total

    # Select action
    if temperature < 0.01:
        action = int(np.argmax(action_probs))
    else:
        probs = action_probs ** (1.0 / temperature)
        prob_sum = probs.sum()
        if prob_sum > 0:
            probs /= prob_sum
            action = int(np.random.choice(NUM_ACTIONS, p=probs))
        else:
            action = int(np.random.choice(legal_actions))

    return action, action_probs
