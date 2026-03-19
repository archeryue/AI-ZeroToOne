"""Candidate 4: Mini AlphaZero — single-process batched multi-game MCTS.

Architecture (single process, zero serialization):
  Run N_PARALLEL games simultaneously. Each MCTS step:
  1. All games traverse their trees, collect leaves (with virtual loss)
  2. ALL leaves from ALL games batched into ONE GPU forward pass
  3. Results scattered back, trees updated

  This eliminates multiprocessing queue overhead entirely.
  C++ engine handles game simulation; GPU handles NN inference.
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
from env.action_space import decode_action, NUM_ACTIONS

# ------ Hyperparameters ------
NUM_BLOCKS = 5
CHANNELS = 64
NUM_SIMULATIONS = 200       # deep search enabled by C++ engine
VIRTUAL_LOSS_N = 16         # leaves per MCTS batch per game
C_PUCT = 1.5
DIRICHLET_ALPHA = 0.3
DIRICHLET_EPSILON = 0.25
TEMP_THRESHOLD = 15
MAX_GAME_STEPS = 200
N_PARALLEL = 16             # games running simultaneously
LR = 2e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 256
REPLAY_BUFFER_SIZE = 100_000
MIN_BUFFER_SIZE = 512
TRAIN_STEPS_PER_ITER = 100
GAMES_PER_ITER = 16         # collect this many games before training
NUM_ITERATIONS = 300
EVAL_EVERY = 25
EVAL_GAMES = 10
SAVE_DIR = os.path.join(SCRIPT_DIR, "candidate4")
# ------------------------------


# ============================================================
# MCTS Node (same as before, but inlined for self-containment)
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
    from env.action_space import encode_move
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
# Batched GPU inference
# ============================================================

def batch_evaluate(network, device, obs_list, mask_list):
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
    return policies, values


# ============================================================
# Multi-game MCTS with cross-game leaf batching
# ============================================================

class GameSlot:
    """One concurrent game being played."""
    def __init__(self):
        self.game = _new_game()
        self.root = None
        self.game_states = {}
        self.examples = []
        self.step = 0
        self.sims_done = 0
        self.finished = False
        self.result = None


def run_self_play_batch(network, device, n_parallel=N_PARALLEL,
                        num_sims=NUM_SIMULATIONS):
    """Play n_parallel games simultaneously with cross-game MCTS batching.

    Returns list of (training_data, result, num_steps) tuples.
    """
    total_evals = 0
    total_batches = 0

    slots = [GameSlot() for _ in range(n_parallel)]
    completed = []

    while True:
        # Restart finished slots until we have enough completed games
        for s in slots:
            if s.finished:
                completed.append((s.examples, s.result, s.step))
                s.__init__()

        if len(completed) >= n_parallel:
            break

        # ---- Phase 1: For each active game, do one MCTS move step ----
        # Find games that need a new root (start of MCTS for a new move)
        needs_root = []
        for s in slots:
            if s.finished or not _is_playing(s.game) or s.step >= MAX_GAME_STEPS:
                if not s.finished:
                    s.result = _game_result(s.game)
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

        # ---- Phase 2: Run MCTS simulations in batches across all games ----
        active_slots = [s for s in slots
                        if not s.finished and s.root is not None
                        and s.sims_done < num_sims]

        while active_slots:
            # Collect leaves from all active games
            pending = []  # (slot, node, game_state)

            for s in active_slots:
                n_leaves = min(VIRTUAL_LOSS_N, num_sims - s.sims_done)

                for _ in range(n_leaves):
                    node = s.root
                    current_game = s.game

                    # Selection
                    while not node.is_leaf():
                        node = node.select_child(C_PUCT)
                        if id(node) in s.game_states:
                            current_game = s.game_states[id(node)]
                        else:
                            current_game = _simulate(current_game, node.action)
                            s.game_states[id(node)] = current_game

                    # Terminal?
                    if not _is_playing(current_game):
                        node.backup(_terminal_value(current_game))
                        s.sims_done += 1
                        continue

                    # Apply virtual loss
                    node.visit_count += 1
                    node.value_sum -= 1.0
                    pending.append((s, node, current_game))
                    s.sims_done += 1

            # Batch evaluate ALL leaves from ALL games
            if pending:
                obs_batch = [_get_obs(g) for _, _, g in pending]
                mask_batch = [_get_mask(g) for _, _, g in pending]
                policies, values = batch_evaluate(network, device,
                                                  obs_batch, mask_batch)
                total_evals += len(obs_batch)
                total_batches += 1

                for k, (s, node, current_game) in enumerate(pending):
                    # Remove virtual loss
                    node.visit_count -= 1
                    node.value_sum += 1.0
                    # Expand
                    legal = _get_legal_actions(current_game)
                    if legal:
                        node.expand(policies[k], legal)
                    # Backup
                    node.backup(values[k])

            # Update active list
            active_slots = [s for s in slots
                            if not s.finished and s.root is not None
                            and s.sims_done < num_sims]

        # ---- Phase 3: Select actions for completed MCTS searches ----
        for s in slots:
            if s.finished or s.root is None:
                continue
            if not s.root.children:
                s.result = _game_result(s.game)
                s.finished = True
                continue

            # Build action probability distribution
            visits = np.array([c.visit_count for c in s.root.children],
                              dtype=np.float32)
            actions = [c.action for c in s.root.children]

            temp = 1.0 if s.step < TEMP_THRESHOLD else 0.1
            if temp < 0.5:
                # Near-greedy
                best = np.argmax(visits)
                action_idx = best
            else:
                visits_temp = visits ** (1.0 / temp)
                probs = visits_temp / visits_temp.sum()
                action_idx = np.random.choice(len(actions), p=probs)

            action = actions[action_idx]

            # Store training example
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
            s.root = None  # new MCTS search next iteration

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


def evaluate_vs_random(network, device, num_games=EVAL_GAMES):
    """Evaluate network vs random. Uses single-game MCTS."""
    from agents.alphazero.mcts import MCTS
    mcts = MCTS(network, device, num_simulations=50, c_puct=C_PUCT)
    wins, losses, draws = 0, 0, 0

    for _ in range(num_games):
        game = _new_game()
        step = 0
        while _is_playing(game) and step < MAX_GAME_STEPS:
            if game.current_turn == 1:  # RED = our agent
                action, _ = mcts.select_action(game, temperature=0.1,
                                               add_noise=False)
            else:
                mask = _get_mask(game)
                legal = np.where(mask)[0]
                action = int(np.random.choice(legal))
            fr, fc, tr, tc = decode_action(action)
            game.make_move(fr, fc, tr, tc)
            step += 1

        result = _game_result(game)
        if result == "red":
            wins += 1
        elif result == "black":
            losses += 1
        else:
            draws += 1

    return wins, losses, draws


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=== Candidate 4: Mini AlphaZero (Single-Process Batched) ===")
    print(f"Device: {device}")

    network = AlphaZeroNet(num_blocks=NUM_BLOCKS, channels=CHANNELS).to(device)
    num_params = sum(p.numel() for p in network.parameters())
    print(f"Model: {NUM_BLOCKS} res blocks, {CHANNELS} channels, "
          f"{num_params:,} params ({num_params * 4 / 1e6:.1f} MB)")
    print(f"MCTS: {NUM_SIMULATIONS} sims, {VIRTUAL_LOSS_N} VL batch, "
          f"c_puct={C_PUCT}")
    print(f"Self-play: {N_PARALLEL} parallel games, "
          f"{MAX_GAME_STEPS} max steps, {GAMES_PER_ITER} games/iter")
    print(f"Training: {NUM_ITERATIONS} iters, "
          f"{TRAIN_STEPS_PER_ITER} train steps/iter")
    print()

    optimizer = Adam(network.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE)

    best_wins = 0
    total_games = 0
    cumulative_evals = 0
    cumulative_batches = 0
    total_start = time.time()

    for iteration in range(1, NUM_ITERATIONS + 1):
        iter_start = time.time()

        # --- Self-play ---
        completed, evals, batches = run_self_play_batch(
            network, device, n_parallel=N_PARALLEL,
            num_sims=NUM_SIMULATIONS)
        cumulative_evals += evals
        cumulative_batches += batches

        selfplay_time = time.time() - iter_start

        # Process completed games
        iter_results = {"red": 0, "black": 0, "draw": 0}
        iter_steps = 0
        RED_VAL = cc.RED if _USE_CPP else 1
        BLACK_VAL = cc.BLACK if _USE_CPP else -1

        for examples, result, steps in completed:
            iter_results[result] += 1
            iter_steps += steps
            total_games += 1
            for obs, action_probs, player in examples:
                if result == "red":
                    value = 1.0 if player == RED_VAL else -1.0
                elif result == "black":
                    value = 1.0 if player == BLACK_VAL else -1.0
                else:
                    value = 0.0
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
        avg_batch = cumulative_evals / max(cumulative_batches, 1)
        print(f"Iter {iteration:3d} | Games: {total_games:5d} | "
              f"R/B/D: {r}/{b}/{d} | Len: {avg_len:.0f} | "
              f"PL: {avg_pl:.4f} | VL: {avg_vl:.4f} | "
              f"Buf: {len(replay_buffer):6d} | "
              f"SP: {selfplay_time:.0f}s ({games_per_min:.1f}g/m) | "
              f"Train: {train_time:.0f}s | Elapsed: {elapsed:.0f}s | "
              f"GPU: {cumulative_evals} evals, avg_batch={avg_batch:.0f}")

        # --- Evaluation ---
        if iteration % EVAL_EVERY == 0:
            wins, losses, draws = evaluate_vs_random(network, device)
            score = (wins + 0.5 * draws) / max(wins + losses + draws, 1) * 100
            print(f"  >> Eval vs Random ({EVAL_GAMES}g as Red): "
                  f"{wins}W / {losses}L / {draws}D  (Score: {score:.0f}%)")
            if wins > best_wins:
                best_wins = wins
                torch.save(network.state_dict(),
                           os.path.join(SAVE_DIR, "az_best.pt"))
                print(f"  >> New best model saved! ({wins}W)")

        # --- Checkpoint ---
        if iteration % 10 == 0:
            torch.save({
                'iteration': iteration,
                'model_state': network.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'total_games': total_games,
                'best_wins': best_wins,
            }, os.path.join(SAVE_DIR, "az_checkpoint.pt"))

    torch.save(network.state_dict(), os.path.join(SAVE_DIR, "az_final.pt"))

    total_time = time.time() - total_start
    print(f"\nTraining complete!")
    print(f"Total games: {total_games}")
    print(f"Best wins vs random: {best_wins}")
    print(f"GPU evals: {cumulative_evals}, batches: {cumulative_batches}")
    print(f"Avg batch size: {cumulative_evals / max(cumulative_batches, 1):.1f}")
    print(f"Time: {total_time / 60:.1f} min ({total_time / 3600:.1f} hrs)")


if __name__ == "__main__":
    main()
