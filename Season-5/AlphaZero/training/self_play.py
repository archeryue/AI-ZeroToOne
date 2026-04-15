"""Self-play game generation using C++ MCTS + PyTorch neural network.

Each game:
  1. Create a fresh Game and MCTSTree
  2. For each move:
     a. Run MCTS simulations (C++ tree search, Python NN eval)
     b. Get policy from visit counts (with temperature)
     c. Sample action from policy
     d. Record (obs, policy, _) — value filled in after game ends
     e. Advance tree (reuse subtree)
  3. After game ends, fill in values (+1 winner, -1 loser)
  4. Push all positions to replay buffer with 8-fold augmentation
"""

import random
import numpy as np
import torch

import go_engine

from model.config import ModelConfig, TrainingConfig
from training.replay_buffer import ReplayBuffer

# Map board size → engine classes
GAME_CLASS = {9: go_engine.Game9, 13: go_engine.Game13, 19: go_engine.Game19}
MCTS_CLASS = {9: go_engine.MCTSTree9, 13: go_engine.MCTSTree13, 19: go_engine.MCTSTree19}


def make_evaluator(net: torch.nn.Module, device: torch.device):
    """Create an NN evaluator callable for C++ MCTS.

    The evaluator takes a numpy obs batch and returns (policies, values).
    """
    def evaluator(obs_batch_np):
        obs = torch.from_numpy(np.array(obs_batch_np)).to(device)
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            # Net returns (policy_logits, value, ownership_logits) since
            # run4; MCTS only needs policy + value.
            logits, values, _own = net(obs)
            policies = torch.softmax(logits, dim=-1)
        return policies.cpu().numpy(), values.cpu().numpy()
    return evaluator


def play_one_game(
    net: torch.nn.Module,
    device: torch.device,
    model_cfg: ModelConfig,
    train_cfg: TrainingConfig,
    game_id: int = 0,
) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """Play a single self-play game. Returns list of (obs, policy, value)."""

    N = model_cfg.board_size
    GameClass = GAME_CLASS[N]
    MCTSClass = MCTS_CLASS[N]

    game = GameClass(train_cfg.komi)
    tree = MCTSClass(
        game,
        train_cfg.c_puct,
        train_cfg.dirichlet_alpha,
        train_cfg.dirichlet_epsilon,
    )

    evaluator = make_evaluator(net, device)
    positions = []  # (obs, policy, current_turn)

    # Resign tracking
    disable_resign = random.random() < train_cfg.resign_disabled_frac
    consecutive_low = 0

    seed = random.randint(0, 2**31)

    for move_num in range(train_cfg.max_game_moves):
        if game.status != go_engine.PLAYING:
            break

        # Temperature schedule
        if move_num < train_cfg.temperature_moves:
            temperature = train_cfg.temperature_high
        else:
            temperature = train_cfg.temperature_low

        # Run MCTS simulations
        tree.run_simulations(
            train_cfg.num_simulations,
            train_cfg.virtual_loss_batch,
            evaluator,
            add_noise=(move_num == 0 or not tree.root_visit_count),
            seed=seed + move_num,
        )

        # Get observation before making the move
        obs = np.array(game.to_observation())

        # Get policy from visit counts
        policy = np.array(tree.get_policy(temperature))

        # Record position
        positions.append((obs, policy, game.current_turn))

        # Resign check
        if not disable_resign:
            root_v = tree.root_value
            if root_v < train_cfg.resign_threshold:
                consecutive_low += 1
            else:
                consecutive_low = 0

            if consecutive_low >= train_cfg.resign_consecutive:
                game.resign(game.current_turn)
                break

        # Sample action from policy
        action = np.random.choice(len(policy), p=policy)

        # Make move in game
        if action == N * N:
            game.pass_move()
        else:
            row, col = action // N, action % N
            game.make_move(row, col)

        # Advance MCTS tree (reuse subtree)
        tree.advance(action)

    # If game didn't end naturally, force end by passing
    while game.status == go_engine.PLAYING:
        game.pass_move()

    # Determine winner: +1 for black win, -1 for white win
    if game.status == go_engine.BLACK_WIN:
        game_result = 1.0
    else:
        game_result = -1.0

    # Per-cell ownership at game end (Tromp-Taylor, absolute frame).
    # Same source as the C++ worker — the auxiliary supervision target
    # for the run4 ownership head. See PHASE_TWO_TRAINING.md.
    abs_ownership = np.array(game.compute_ownership(), dtype=np.int8)

    # Fill in values + per-record ownership in current-player perspective.
    samples = []
    for obs, policy, turn in positions:
        if turn == go_engine.BLACK:
            value = game_result
            persp = 1
        else:
            value = -game_result
            persp = -1
        ownership = (abs_ownership * persp).astype(np.int8)
        samples.append((obs, policy, value, ownership))

    return samples


def run_self_play(
    net: torch.nn.Module,
    device: torch.device,
    model_cfg: ModelConfig,
    train_cfg: TrainingConfig,
    buffer: ReplayBuffer,
    num_games: int,
) -> dict:
    """Run multiple self-play games and add positions to replay buffer.

    Returns stats dict with game counts, positions, results.
    """
    net.eval()

    total_positions = 0
    black_wins = 0
    white_wins = 0
    total_moves = 0

    for i in range(num_games):
        samples = play_one_game(net, device, model_cfg, train_cfg, game_id=i)

        for obs, policy, value, ownership in samples:
            buffer.push(obs, policy, value, ownership)

        total_positions += len(samples)
        total_moves += len(samples)

        # Track results (value is from perspective of player-to-move)
        if samples and samples[0][2] > 0:
            black_wins += 1
        else:
            white_wins += 1

    return {
        "games": num_games,
        "positions": total_positions,
        "positions_augmented": total_positions * 8,
        "avg_moves": total_moves / max(num_games, 1),
        "black_wins": black_wins,
        "white_wins": white_wins,
        "buffer_size": len(buffer),
    }
