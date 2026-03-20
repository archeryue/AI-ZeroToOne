"""Candidate 4 v5: AlphaZero with Curriculum Training.

Architecture (single process, zero serialization):
  Run N_PARALLEL games simultaneously. Each MCTS step:
  1. All games traverse their trees, collect leaves (with virtual loss)
  2. ALL leaves from ALL games batched into ONE GPU forward pass
  3. Results scattered back, trees updated

  Curriculum: Random → Greedy → Minimax → Self-play
  Promote to next opponent when eval score ≥ 75%.
"""

import os
import sys
import time
import random
import math
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

# --- Path setup ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
CHESS_DIR = os.path.join(os.path.dirname(PROJECT_DIR), "ChineseChess", "backend")

# Import C++ engine BEFORE adding project paths (avoid local engine_c/ shadowing)
try:
    import engine_c as cc
    _USE_CPP = True
    print("[engine] Using C++ engine")
except ImportError:
    _USE_CPP = False
    print("[engine] WARNING: C++ engine not found, using Python (slow)")

sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, CHESS_DIR)

from agents.alphazero.network import AlphaZeroNet
from env.action_space import decode_action, encode_move, NUM_ACTIONS

# ------ Hyperparameters ------
NUM_BLOCKS = 5
CHANNELS = 64
NUM_SIMULATIONS = 200       # more sims → better policy targets from MCTS
VIRTUAL_LOSS_N = 8          # leaves per MCTS batch per game
C_PUCT = 1.5
DIRICHLET_ALPHA = 0.3
DIRICHLET_EPSILON = 0.25
TEMP_THRESHOLD = 30         # explore more moves with temperature
MAX_GAME_STEPS = 200
N_PARALLEL = 16             # games running simultaneously
LR = 5e-4
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 256
REPLAY_BUFFER_SIZE = 20_000
MIN_BUFFER_SIZE = 1024
TRAIN_STEPS_PER_ITER = 100
GAMES_PER_ITER = 16
NUM_ITERATIONS = 500
EVAL_EVERY = 10
EVAL_GAMES = 10
SAVE_DIR = os.path.join(SCRIPT_DIR, "candidate4_v5")
# --- Curriculum ---
CURRICULUM_PHASES = ["random", "greedy", "minimax", "self_play"]
PROMOTE_THRESHOLD = 0.75    # 75% score vs current opponent to promote
# ------------------------------


# ============================================================
# MCTS Node
# ============================================================

class MCTSNode:
    __slots__ = ['parent', 'action', 'prior', 'visit_count', 'value_sum',
                 'children', 'is_expanded']

    def __init__(self, parent=None, action=None, prior=0.0):
        self.parent = parent
        self.action = action
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0.0
        self.children = []
        self.is_expanded = False

    def q_value(self):
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def select_child(self, c_puct):
        total_visits = sum(c.visit_count for c in self.children)
        sqrt_total = math.sqrt(total_visits + 1)

        best_score = -float('inf')
        best_child = None
        for child in self.children:
            q = child.q_value()
            u = c_puct * child.prior * sqrt_total / (1 + child.visit_count)
            score = q + u
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    def expand(self, policy, legal_actions):
        self.is_expanded = True
        total_p = sum(policy[a] for a in legal_actions) + 1e-8
        for a in legal_actions:
            self.children.append(MCTSNode(parent=self, action=a,
                                          prior=policy[a] / total_p))

    def backup(self, value):
        node = self
        while node is not None:
            node.visit_count += 1
            node.value_sum += value
            value = -value
            node = node.parent

    def is_leaf(self):
        return not self.is_expanded


# ============================================================
# Game state helpers (C++ or Python)
# ============================================================

def _new_game():
    if _USE_CPP:
        return cc.Game()
    from engine.game import Game
    return Game()


def _is_playing(game):
    if _USE_CPP:
        return game.status == cc.STATUS_PLAYING
    from engine.game import GameStatus
    return game.status == GameStatus.PLAYING


def _simulate(game, action):
    if _USE_CPP:
        return game.simulate_action(action)
    fr, fc, tr, tc = decode_action(action)
    from engine.game import Game
    new_game = Game.__new__(Game)
    new_game.board = game.board.copy()
    new_game.current_turn = game.current_turn
    new_game.move_history = []
    new_game.board_history = []
    new_game.status = game.status
    piece = new_game.board.get(fr, fc)
    new_game.board.set(fr, fc, 0)
    new_game.board.set(tr, tc, piece)
    new_game.current_turn = -new_game.current_turn
    new_game._check_game_over()
    return new_game


def _get_obs(game):
    if _USE_CPP:
        return cc.board_to_observation(game.board, game.current_turn)
    from env.observation import board_to_observation
    return board_to_observation(game.board, game.current_turn)


def _get_mask(game):
    if _USE_CPP:
        return cc.get_action_mask(game.board, game.current_turn)
    from env.action_space import get_action_mask
    return get_action_mask(game.board, game.current_turn)


def _get_legal_actions(game):
    if _USE_CPP:
        return cc.get_legal_action_indices(game.board, game.current_turn)
    from engine.rules import get_legal_moves
    return [encode_move(m.from_row, m.from_col, m.to_row, m.to_col)
            for m in get_legal_moves(game.board, game.current_turn)]


def _terminal_value(game):
    """Value from perspective of current player (who needs to move)."""
    if _USE_CPP:
        if game.status == cc.STATUS_RED_WIN:
            return 1.0 if game.current_turn == cc.RED else -1.0
        elif game.status == cc.STATUS_BLACK_WIN:
            return 1.0 if game.current_turn == cc.BLACK else -1.0
        return 0.0
    from engine.game import GameStatus
    from engine.board import RED, BLACK
    if game.status == GameStatus.RED_WIN:
        return 1.0 if game.current_turn == RED else -1.0
    elif game.status == GameStatus.BLACK_WIN:
        return 1.0 if game.current_turn == BLACK else -1.0
    return 0.0


def _game_result(game):
    if _USE_CPP:
        if game.status == cc.STATUS_RED_WIN:
            return "red"
        elif game.status == cc.STATUS_BLACK_WIN:
            return "black"
        return "draw"
    from engine.game import GameStatus
    if game.status == GameStatus.RED_WIN:
        return "red"
    elif game.status == GameStatus.BLACK_WIN:
        return "black"
    return "draw"


# ============================================================
# Material evaluation
# ============================================================

_PIECE_VALUES = {1: 0, 2: 2, 3: 2, 4: 4, 5: 9, 6: 4.5, 7: 1}


def _material_value(game):
    """Return value in [-1, 1] from Red's perspective."""
    red_mat, black_mat = 0.0, 0.0
    for r in range(10):
        for c in range(9):
            piece = game.board.get(r, c)
            if piece > 0:
                red_mat += _PIECE_VALUES.get(piece, 0)
            elif piece < 0:
                black_mat += _PIECE_VALUES.get(-piece, 0)
    diff = red_mat - black_mat
    return max(-1.0, min(1.0, diff / 10.0))


MATERIAL_BLEND = 0.3
CHECK_BONUS = 0.15          # bonus when opponent's general is in check
ENDGAME_START_PROB = 0.25   # probability of starting from a random endgame position


def _in_check(game):
    """Check if the current player's general is under attack."""
    turn = game.current_turn
    is_red = (turn == cc.RED) if _USE_CPP else (turn == 1)
    gen_val = 1 if is_red else -1

    # Find general position
    gr, gc = -1, -1
    for r in range(10):
        for c in range(9):
            if game.board.get(r, c) == gen_val:
                gr, gc = r, c
                break
        if gr >= 0:
            break
    if gr < 0:
        return False

    # Check attacks from opponent pieces
    for r in range(10):
        for c in range(9):
            p = game.board.get(r, c)
            if p == 0 or (p > 0) == (gen_val > 0):
                continue
            ap = abs(p)

            if ap == 5:  # Chariot: same row/col, no blockers
                if r == gr and c != gc:
                    lo, hi = min(c, gc) + 1, max(c, gc)
                    if all(game.board.get(r, i) == 0 for i in range(lo, hi)):
                        return True
                elif c == gc and r != gr:
                    lo, hi = min(r, gr) + 1, max(r, gr)
                    if all(game.board.get(i, c) == 0 for i in range(lo, hi)):
                        return True

            elif ap == 6:  # Cannon: same row/col, exactly 1 blocker
                if r == gr and c != gc:
                    lo, hi = min(c, gc) + 1, max(c, gc)
                    if sum(1 for i in range(lo, hi) if game.board.get(r, i) != 0) == 1:
                        return True
                elif c == gc and r != gr:
                    lo, hi = min(r, gr) + 1, max(r, gr)
                    if sum(1 for i in range(lo, hi) if game.board.get(i, c) != 0) == 1:
                        return True

            elif ap == 4:  # Horse: L-shape with leg check
                dr, dc = gr - r, gc - c
                if (abs(dr), abs(dc)) in [(2, 1), (1, 2)]:
                    if abs(dr) == 2:
                        if game.board.get(r + dr // 2, c) == 0:
                            return True
                    else:
                        if game.board.get(r, c + dc // 2) == 0:
                            return True

            elif ap == 7:  # Soldier: attacks forward or sideways (if past river)
                if gen_val > 0:  # Red general, attacked by black soldier (p < 0)
                    # Black soldiers move down (increasing row) before river, any dir after
                    if r + 1 == gr and c == gc:  # forward attack
                        return True
                    if r >= 5 and r == gr and abs(c - gc) == 1:  # sideways past river
                        return True
                else:  # Black general, attacked by red soldier (p > 0)
                    if r - 1 == gr and c == gc:
                        return True
                    if r <= 4 and r == gr and abs(c - gc) == 1:
                        return True

    return False


def _material_value_for_player(game):
    """Material advantage from current player's perspective, in [-1, 1]."""
    red_mat, black_mat = 0.0, 0.0
    for r in range(10):
        for c in range(9):
            piece = game.board.get(r, c)
            if piece > 0:
                red_mat += _PIECE_VALUES.get(piece, 0)
            elif piece < 0:
                black_mat += _PIECE_VALUES.get(-piece, 0)
    diff = red_mat - black_mat
    turn = game.current_turn
    if _USE_CPP:
        player_diff = diff if turn == cc.RED else -diff
    else:
        player_diff = diff if turn == 1 else -diff
    return max(-1.0, min(1.0, player_diff / 10.0))


# ============================================================
# Curriculum opponents
# ============================================================

def _opponent_random(game):
    """Pick a random legal action."""
    legal = _get_legal_actions(game)
    return random.choice(legal)


def _opponent_greedy(game):
    """Pick the action that maximizes material advantage (1-ply lookahead)."""
    legal = _get_legal_actions(game)
    is_red = (game.current_turn == cc.RED) if _USE_CPP else (game.current_turn == 1)

    best_score = -float('inf')
    best_actions = []
    for action in legal:
        child = _simulate(game, action)
        mat = _material_value(child)  # from Red's perspective
        score = mat if is_red else -mat
        if score > best_score:
            best_score = score
            best_actions = [action]
        elif abs(score - best_score) < 1e-8:
            best_actions.append(action)

    return random.choice(best_actions)


def _minimax_search(game, depth, alpha, beta, maximizing_red):
    """Alpha-beta minimax. Returns eval from Red's perspective."""
    if depth == 0 or not _is_playing(game):
        return _material_value(game)

    legal = _get_legal_actions(game)
    if not legal:
        return _material_value(game)

    if maximizing_red:
        value = -float('inf')
        for action in legal:
            child = _simulate(game, action)
            value = max(value, _minimax_search(child, depth - 1, alpha, beta, False))
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return value
    else:
        value = float('inf')
        for action in legal:
            child = _simulate(game, action)
            value = min(value, _minimax_search(child, depth - 1, alpha, beta, True))
            beta = min(beta, value)
            if alpha >= beta:
                break
        return value


def _opponent_minimax(game, depth=2):
    """Pick the best action using alpha-beta minimax search."""
    legal = _get_legal_actions(game)
    is_red = (game.current_turn == cc.RED) if _USE_CPP else (game.current_turn == 1)

    best_score = -float('inf')
    best_actions = []
    for action in legal:
        child = _simulate(game, action)
        # Search from opponent's perspective after the move
        score = _minimax_search(child, depth - 1, -float('inf'), float('inf'),
                                not is_red)
        # Flip so higher = better for current player
        if not is_red:
            score = -score
        if score > best_score:
            best_score = score
            best_actions = [action]
        elif abs(score - best_score) < 1e-8:
            best_actions.append(action)

    return random.choice(best_actions)


def _make_endgame_position():
    """Create a random mid/endgame position by playing random moves."""
    game = _new_game()
    n_moves = random.randint(40, 80)
    for _ in range(n_moves):
        if not _is_playing(game):
            return _new_game()  # game ended, fall back to start
        legal = _get_legal_actions(game)
        if not legal:
            return _new_game()
        action = random.choice(legal)
        fr, fc, tr, tc = decode_action(action)
        game.make_move(fr, fc, tr, tc)
    if not _is_playing(game):
        return _new_game()
    return game


def _get_opponent_move(game, opponent_type):
    """Get opponent's move based on opponent type."""
    if opponent_type == "random":
        return _opponent_random(game)
    elif opponent_type == "greedy":
        return _opponent_greedy(game)
    elif opponent_type == "minimax":
        return _opponent_minimax(game, depth=2)
    else:
        raise ValueError(f"Unknown opponent type: {opponent_type}")


# ============================================================
# Batched GPU inference
# ============================================================

def batch_evaluate(network, device, obs_list, mask_list, game_list=None):
    """Single batched GPU forward pass. Returns (policies, values) numpy."""
    if not obs_list:
        return np.empty((0, NUM_ACTIONS)), np.empty(0)
    network.eval()
    with torch.no_grad():
        obs_t = torch.from_numpy(np.array(obs_list)).to(device)
        mask_t = torch.from_numpy(np.array(mask_list)).to(device)
        log_p, v = network(obs_t, mask_t)
        policies = torch.exp(log_p).cpu().numpy()
        values = v.cpu().numpy()

    if game_list is not None:
        for i, g in enumerate(game_list):
            if MATERIAL_BLEND > 0:
                mat_v = _material_value_for_player(g)
                values[i] = (1 - MATERIAL_BLEND) * values[i] + MATERIAL_BLEND * mat_v
            # Check bonus: if current player is in check, that's bad for them
            # (good for the side that delivered check)
            if CHECK_BONUS > 0 and _in_check(g):
                values[i] -= CHECK_BONUS

    return policies, values


# ============================================================
# Multi-game MCTS with cross-game leaf batching
# ============================================================

RED_VAL = None  # set in main()
BLACK_VAL = None


class GameSlot:
    """One concurrent game being played."""
    def __init__(self, agent_color=None, use_endgame=False):
        if use_endgame and random.random() < ENDGAME_START_PROB:
            self.game = _make_endgame_position()
        else:
            self.game = _new_game()
        self.root = None
        self.game_states = {}
        self.examples = []
        self.step = 0
        self.sims_done = 0
        self.finished = False
        self.result = None
        self.draw_reason = None
        self.position_history = {}
        self.agent_color = agent_color
        self.use_endgame = use_endgame


def _is_agent_turn(slot):
    """Check if it's the agent's turn (True for self-play always)."""
    if slot.agent_color is None:
        return True  # self-play: both sides are the agent
    return slot.game.current_turn == slot.agent_color


def run_self_play_batch(network, device, n_parallel=N_PARALLEL,
                        num_sims=NUM_SIMULATIONS, opponent_type="self_play"):
    """Play n_parallel games with cross-game MCTS batching.

    For curriculum phases, agent plays one side and opponent plays the other.
    Only agent's positions (with MCTS policy) are stored as training examples.
    """
    total_evals = 0
    total_batches = 0

    # Assign agent colors: half Red, half Black (None for self-play)
    if opponent_type == "self_play":
        slots = [GameSlot(agent_color=None, use_endgame=True)
                 for _ in range(n_parallel)]
    else:
        slots = []
        for i in range(n_parallel):
            color = RED_VAL if i < n_parallel // 2 else BLACK_VAL
            slots.append(GameSlot(agent_color=color, use_endgame=True))

    completed = []

    while True:
        # Restart finished slots
        for s in slots:
            if s.finished:
                completed.append((s.examples, s.result, s.step,
                                  s.draw_reason, s.game, s.agent_color))
                old_color = s.agent_color
                old_endgame = s.use_endgame
                s.__init__(agent_color=old_color, use_endgame=old_endgame)

        if len(completed) >= n_parallel:
            break

        # ---- Handle opponent turns first (no MCTS needed) ----
        for s in slots:
            if s.finished or not _is_playing(s.game) or s.step >= MAX_GAME_STEPS:
                continue
            # Play opponent moves until it's the agent's turn (or game ends)
            while (not _is_agent_turn(s) and _is_playing(s.game)
                   and s.step < MAX_GAME_STEPS):
                action = _get_opponent_move(s.game, opponent_type)
                fr, fc, tr, tc = decode_action(action)
                s.game.make_move(fr, fc, tr, tc)
                s.step += 1
                fen = s.game.board.to_fen()
                s.position_history[fen] = s.position_history.get(fen, 0) + 1

        # ---- Phase 1: Check game endings and set up MCTS roots ----
        needs_root = []
        for s in slots:
            if s.finished or not _is_playing(s.game) or s.step >= MAX_GAME_STEPS:
                if not s.finished:
                    result = _game_result(s.game)
                    if result != "draw":
                        s.result = result
                    elif s.step >= MAX_GAME_STEPS:
                        # Material adjudication: need significant lead (~Horse/Cannon)
                        mat = _material_value(s.game)
                        if mat > 0.3:
                            s.result = "red"
                        elif mat < -0.3:
                            s.result = "black"
                        else:
                            s.result = "draw"
                        s.draw_reason = "step_limit"
                    else:
                        s.result = "draw"
                        s.draw_reason = "stalemate"
                    s.finished = True
                continue
            if s.root is None:
                needs_root.append(s)

        # Batch-evaluate roots
        if needs_root:
            obs_batch = [_get_obs(s.game) for s in needs_root]
            mask_batch = [_get_mask(s.game) for s in needs_root]
            policies, values = batch_evaluate(network, device,
                                              obs_batch, mask_batch)
            total_evals += len(obs_batch)
            total_batches += 1

            for i, s in enumerate(needs_root):
                s.root = MCTSNode()
                s.game_states = {id(s.root): s.game}
                legal = _get_legal_actions(s.game)
                if not legal:
                    s.result = _game_result(s.game)
                    s.finished = True
                    continue
                s.root.expand(policies[i], legal)
                # Dirichlet noise
                if s.step < TEMP_THRESHOLD and s.root.children:
                    noise = np.random.dirichlet(
                        [DIRICHLET_ALPHA] * len(s.root.children))
                    for j, child in enumerate(s.root.children):
                        child.prior = ((1 - DIRICHLET_EPSILON) * child.prior
                                       + DIRICHLET_EPSILON * noise[j])
                s.sims_done = 0

        # ---- Phase 2: Run MCTS simulations in batches ----
        active_slots = [s for s in slots
                        if not s.finished and s.root is not None
                        and s.sims_done < num_sims]

        while active_slots:
            pending = []

            for s in active_slots:
                n_leaves = min(VIRTUAL_LOSS_N, num_sims - s.sims_done)

                for _ in range(n_leaves):
                    node = s.root
                    current_game = s.game

                    while not node.is_leaf():
                        node = node.select_child(C_PUCT)
                        if id(node) in s.game_states:
                            current_game = s.game_states[id(node)]
                        else:
                            current_game = _simulate(current_game, node.action)
                            s.game_states[id(node)] = current_game

                    if not _is_playing(current_game):
                        node.backup(_terminal_value(current_game))
                        s.sims_done += 1
                        continue

                    node.visit_count += 1
                    node.value_sum -= 1.0
                    pending.append((s, node, current_game))
                    s.sims_done += 1

            if pending:
                obs_batch = [_get_obs(g) for _, _, g in pending]
                mask_batch = [_get_mask(g) for _, _, g in pending]
                game_batch = [g for _, _, g in pending]
                policies, values = batch_evaluate(network, device,
                                                  obs_batch, mask_batch,
                                                  game_batch)
                total_evals += len(obs_batch)
                total_batches += 1

                for k, (s, node, current_game) in enumerate(pending):
                    node.visit_count -= 1
                    node.value_sum += 1.0
                    legal = _get_legal_actions(current_game)
                    if legal:
                        node.expand(policies[k], legal)
                    node.backup(values[k])

            active_slots = [s for s in slots
                            if not s.finished and s.root is not None
                            and s.sims_done < num_sims]

        # ---- Phase 3: Select actions from MCTS ----
        for s in slots:
            if s.finished or s.root is None:
                continue
            if not s.root.children:
                s.result = _game_result(s.game)
                s.finished = True
                continue

            visits = np.array([c.visit_count for c in s.root.children],
                              dtype=np.float32)
            actions = [c.action for c in s.root.children]

            # Ban moves that would repeat a position 3+ times
            for i, child in enumerate(s.root.children):
                if id(child) in s.game_states:
                    child_fen = s.game_states[id(child)].board.to_fen()
                else:
                    sim_game = _simulate(s.game, child.action)
                    child_fen = sim_game.board.to_fen()
                if s.position_history.get(child_fen, 0) >= 2:
                    visits[i] = 0.0

            if visits.sum() == 0:
                visits = np.array([c.visit_count for c in s.root.children],
                                  dtype=np.float32)

            temp = 1.0 if s.step < TEMP_THRESHOLD else 0.1
            if temp < 0.5:
                action_idx = np.argmax(visits)
            else:
                visits_temp = visits ** (1.0 / temp)
                probs = visits_temp / visits_temp.sum()
                action_idx = np.random.choice(len(actions), p=probs)

            action = actions[action_idx]

            # Store training example (MCTS policy target)
            action_probs = np.zeros(NUM_ACTIONS, dtype=np.float32)
            visit_sum = visits.sum()
            if visit_sum > 0:
                for a, v in zip(actions, visits):
                    action_probs[a] = v / visit_sum

            obs = _get_obs(s.game)
            s.examples.append((obs, action_probs, s.game.current_turn))

            # Apply move
            fr, fc, tr, tc = decode_action(action)
            s.game.make_move(fr, fc, tr, tc)
            s.step += 1

            fen = s.game.board.to_fen()
            s.position_history[fen] = s.position_history.get(fen, 0) + 1

            s.root = None

    return completed, total_evals, total_batches


# ============================================================
# Training
# ============================================================

def train_step(network, optimizer, batch, device):
    network.train()
    obs_batch, policy_batch, value_batch = zip(*batch)
    obs = torch.from_numpy(np.array(obs_batch)).to(device)
    target_policy = torch.from_numpy(np.array(policy_batch)).to(device)
    target_value = torch.from_numpy(
        np.array(value_batch, dtype=np.float32)).to(device)

    action_mask = target_policy > 1e-8
    log_policy, value = network(obs, action_mask)

    safe_log_policy = log_policy.clone()
    safe_log_policy[log_policy == float('-inf')] = 0.0
    policy_loss = -torch.sum(target_policy * safe_log_policy, dim=-1).mean()

    value_loss = F.mse_loss(value, target_value)
    loss = policy_loss + value_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return policy_loss.item(), value_loss.item()


def evaluate_vs_opponent(network, device, opponent_type, num_games=EVAL_GAMES):
    """Evaluate network vs a curriculum opponent. Agent plays both sides."""
    from agents.alphazero.mcts import MCTS
    mcts = MCTS(network, device, num_simulations=NUM_SIMULATIONS, c_puct=C_PUCT)
    wins, losses, draws = 0, 0, 0

    for game_idx in range(num_games):
        game = _new_game()
        step = 0
        pos_hist = {}
        # Alternate: even games agent=Red, odd games agent=Black
        agent_is_red = (game_idx % 2 == 0)

        while _is_playing(game) and step < MAX_GAME_STEPS:
            is_red_turn = (game.current_turn == (cc.RED if _USE_CPP else 1))
            is_agent_turn = (is_red_turn == agent_is_red)

            if is_agent_turn:
                action, _ = mcts.select_action(game, temperature=0.1,
                                               add_noise=False)
                # Ban repeated moves
                sim = _simulate(game, action)
                sim_fen = sim.board.to_fen()
                if pos_hist.get(sim_fen, 0) >= 2:
                    mask = _get_mask(game)
                    legal = np.where(mask)[0]
                    for alt in legal:
                        alt_sim = _simulate(game, alt)
                        alt_fen = alt_sim.board.to_fen()
                        if pos_hist.get(alt_fen, 0) < 2:
                            action = alt
                            break
            else:
                if opponent_type == "random":
                    mask = _get_mask(game)
                    legal = np.where(mask)[0]
                    action = int(np.random.choice(legal))
                else:
                    action = _get_opponent_move(game, opponent_type)

            fr, fc, tr, tc = decode_action(action)
            game.make_move(fr, fc, tr, tc)
            step += 1
            fen = game.board.to_fen()
            pos_hist[fen] = pos_hist.get(fen, 0) + 1

        result = _game_result(game)
        # Determine win/loss from agent's perspective
        if agent_is_red:
            if result == "red":
                wins += 1
            elif result == "black":
                losses += 1
            else:
                draws += 1
        else:
            if result == "black":
                wins += 1
            elif result == "red":
                losses += 1
            else:
                draws += 1

    return wins, losses, draws


def load_human_positions(data_dir, n_positions=10000):
    """Load random positions from supervised training shards for replay buffer seeding."""
    import glob as glob_mod
    shard_files = sorted(glob_mod.glob(
        os.path.join(data_dir, "supervised_training_data_shard*.npz")))
    if not shard_files:
        print("No supervised data shards found, skipping buffer seeding")
        return []

    shard_file = random.choice(shard_files)
    data = np.load(shard_file)
    boards = data['boards']
    actions = data['actions']
    values = data['values']
    turns = data['turns']

    n = min(n_positions, len(boards))
    indices = np.random.choice(len(boards), n, replace=False)

    positions = []
    for idx in indices:
        grid = boards[idx].reshape(10, 9)
        turn = int(turns[idx])

        if _USE_CPP:
            board = cc.Board(grid.tolist())
            obs = cc.board_to_observation(board, turn)
            mask = cc.get_action_mask(board, turn)
        else:
            from env.observation import board_to_observation
            from engine.board import Board
            from env.action_space import get_action_mask
            board = Board(grid.tolist())
            obs = board_to_observation(board, turn)
            mask = get_action_mask(board, turn)

        action_probs = np.zeros(NUM_ACTIONS, dtype=np.float32)
        action_probs[actions[idx]] = 1.0

        positions.append((obs, action_probs, float(values[idx])))

    print(f"Loaded {len(positions)} human positions from {os.path.basename(shard_file)}")
    return positions


def main():
    global RED_VAL, BLACK_VAL
    RED_VAL = cc.RED if _USE_CPP else 1
    BLACK_VAL = cc.BLACK if _USE_CPP else -1

    os.makedirs(SAVE_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=== Candidate 4 v5: AlphaZero with Curriculum ===")
    print(f"Device: {device}")

    network = AlphaZeroNet(num_blocks=NUM_BLOCKS, channels=CHANNELS).to(device)

    SKIP_PRETRAIN = os.environ.get("SKIP_PRETRAIN", "0") == "1"
    pretrained_path = os.path.join(SAVE_DIR, "az_pretrained.pt")
    if not SKIP_PRETRAIN and os.path.exists(pretrained_path):
        network.load_state_dict(torch.load(pretrained_path, map_location=device))
        print(f"Loaded pretrained weights from {pretrained_path}")
    elif SKIP_PRETRAIN:
        print("Skipping pretrained weights (SKIP_PRETRAIN=1)")

    num_params = sum(p.numel() for p in network.parameters())
    print(f"Model: {NUM_BLOCKS} res blocks, {CHANNELS} channels, "
          f"{num_params:,} params ({num_params * 4 / 1e6:.1f} MB)")
    print(f"MCTS: {NUM_SIMULATIONS} sims, {VIRTUAL_LOSS_N} VL batch, "
          f"c_puct={C_PUCT}")
    print(f"Self-play: {N_PARALLEL} parallel games, "
          f"{MAX_GAME_STEPS} max steps")
    print(f"Training: {NUM_ITERATIONS} iters, "
          f"{TRAIN_STEPS_PER_ITER} train steps/iter")
    print(f"Curriculum: {' → '.join(CURRICULUM_PHASES)}, "
          f"promote at {PROMOTE_THRESHOLD*100:.0f}%")
    print()

    optimizer = Adam(network.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE)

    if not SKIP_PRETRAIN:
        data_dir = os.path.join(PROJECT_DIR, "data")
        human_positions = load_human_positions(data_dir, n_positions=20000)
        for pos in human_positions:
            replay_buffer.append(pos)
        print(f"Replay buffer seeded with {len(replay_buffer)} human positions")
    else:
        print("Skipping human buffer seeding (SKIP_PRETRAIN=1)")

    best_wins = 0
    total_games = 0
    cumulative_evals = 0
    cumulative_batches = 0
    total_start = time.time()

    # Checkmate boost: track running checkmate rate over recent games
    recent_checkmates = deque(maxlen=160)  # ~10 iterations of games

    # Curriculum state
    phase_idx = 0
    current_phase = CURRICULUM_PHASES[phase_idx]

    for iteration in range(1, NUM_ITERATIONS + 1):
        iter_start = time.time()

        # --- Self-play / Curriculum play ---
        completed, evals, batches = run_self_play_batch(
            network, device, n_parallel=N_PARALLEL,
            num_sims=NUM_SIMULATIONS, opponent_type=current_phase)
        cumulative_evals += evals
        cumulative_batches += batches

        selfplay_time = time.time() - iter_start

        # Process completed games
        iter_results = {"red": 0, "black": 0, "draw": 0}
        iter_steps = 0
        iter_checkmates = 0

        for examples, result, steps, draw_reason, final_game, agent_color in completed:
            iter_results[result] += 1
            iter_steps += steps
            total_games += 1
            is_checkmate = (result != "draw" and draw_reason != "step_limit")
            if is_checkmate:
                iter_checkmates += 1
            recent_checkmates.append(1 if is_checkmate else 0)

            # Penalty for long games: -0.005 per step after step 100
            step_penalty = -0.005 * max(0, steps - 100)

            # Half reward (0.5) for material adjudication wins
            reward_scale = 0.5 if draw_reason == "step_limit" else 1.0

            # Checkmate boost: upweight rare checkmate games
            cm_rate = sum(recent_checkmates) / max(len(recent_checkmates), 1)
            if is_checkmate and cm_rate < 0.5:
                boost = max(1, min(10, round(0.5 / max(cm_rate, 0.05))))
            else:
                boost = 1

            for obs, action_probs, player in examples:
                if result == "red":
                    value = reward_scale if player == RED_VAL else -reward_scale
                elif result == "black":
                    value = reward_scale if player == BLACK_VAL else -reward_scale
                else:
                    value = 0.0
                value += step_penalty
                for _ in range(boost):
                    replay_buffer.append((obs, action_probs, value))

        games_per_min = len(completed) / (selfplay_time / 60) if selfplay_time > 0 else 0
        avg_len = iter_steps / max(len(completed), 1)

        # --- Training ---
        train_start = time.time()
        total_pl, total_vl = 0.0, 0.0
        train_steps = 0

        if len(replay_buffer) >= MIN_BUFFER_SIZE:
            network.train()
            for _ in range(TRAIN_STEPS_PER_ITER):
                batch = random.sample(list(replay_buffer),
                                      min(BATCH_SIZE, len(replay_buffer)))
                pl, vl = train_step(network, optimizer, batch, device)
                total_pl += pl
                total_vl += vl
                train_steps += 1

        train_time = time.time() - train_start
        avg_pl = total_pl / max(train_steps, 1)
        avg_vl = total_vl / max(train_steps, 1)

        elapsed = time.time() - total_start
        r, b, d = iter_results["red"], iter_results["black"], iter_results["draw"]
        cm_pct = iter_checkmates / max(len(completed), 1) * 100
        cm_rate = sum(recent_checkmates) / max(len(recent_checkmates), 1)
        cm_boost = max(1, min(10, round(0.5 / max(cm_rate, 0.05)))) if cm_rate < 0.5 else 1
        print(f"Iter {iteration:3d} [{current_phase:>8s}] | Games: {total_games:5d} | "
              f"R/B/D: {r}/{b}/{d} CM:{iter_checkmates}({cm_pct:.0f}%)x{cm_boost} | Len: {avg_len:.0f} | "
              f"PL: {avg_pl:.4f} | VL: {avg_vl:.4f} | "
              f"Buf: {len(replay_buffer):6d} | "
              f"SP: {selfplay_time:.0f}s ({games_per_min:.1f}g/m) | "
              f"Train: {train_time:.0f}s | Elapsed: {elapsed:.0f}s")

        # --- Evaluation ---
        if iteration % EVAL_EVERY == 0:
            # Eval vs current phase opponent for promotion
            wins, losses, draws = evaluate_vs_opponent(
                network, device, current_phase)
            score = (wins + 0.5 * draws) / max(wins + losses + draws, 1) * 100
            print(f"  >> Eval vs {current_phase} ({EVAL_GAMES}g): "
                  f"{wins}W / {losses}L / {draws}D  (Score: {score:.0f}%)")

            if wins > best_wins:
                best_wins = wins
                torch.save(network.state_dict(),
                           os.path.join(SAVE_DIR, "az_best.pt"))
                print(f"  >> New best model saved! ({wins}W)")

            # Check promotion
            if (score / 100 >= PROMOTE_THRESHOLD
                    and phase_idx < len(CURRICULUM_PHASES) - 1):
                phase_idx += 1
                current_phase = CURRICULUM_PHASES[phase_idx]
                print(f"  >> PROMOTED to phase: {current_phase}!")

        # --- Checkpoint ---
        if iteration % 10 == 0:
            torch.save({
                'iteration': iteration,
                'model_state': network.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'total_games': total_games,
                'best_wins': best_wins,
                'phase_idx': phase_idx,
            }, os.path.join(SAVE_DIR, "az_checkpoint.pt"))

    torch.save(network.state_dict(), os.path.join(SAVE_DIR, "az_final.pt"))

    total_time = time.time() - total_start
    print(f"\nTraining complete!")
    print(f"Total games: {total_games}")
    print(f"Best wins: {best_wins}")
    print(f"Final phase: {current_phase}")
    print(f"GPU evals: {cumulative_evals}, batches: {cumulative_batches}")
    print(f"Avg batch size: {cumulative_evals / max(cumulative_batches, 1):.1f}")
    print(f"Time: {total_time / 60:.1f} min ({total_time / 3600:.1f} hrs)")


if __name__ == "__main__":
    main()
