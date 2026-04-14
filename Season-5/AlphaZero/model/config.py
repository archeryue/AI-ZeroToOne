"""Model and training configurations for AlphaZero."""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    board_size: int
    num_blocks: int
    channels: int
    input_planes: int = 17  # 8 history × 2 colors + 1 color-to-play

    @property
    def cells(self) -> int:
        return self.board_size ** 2

    @property
    def actions(self) -> int:
        return self.cells + 1  # board positions + pass


@dataclass
class TrainingConfig:
    # MCTS
    num_simulations: int = 400
    virtual_loss_batch: int = 8
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.11
    dirichlet_epsilon: float = 0.25
    temperature_moves: int = 15  # moves with temp=1.0, then temp→0.1
    temperature_high: float = 1.0
    temperature_low: float = 0.1

    # Self-play
    num_parallel_games: int = 256
    num_games_per_iter: int = 2048
    max_game_moves: int = 200  # max moves before forcing end

    # Resign (v2 — see engine/worker.h SelfPlayConfig comment)
    resign_threshold: float = -0.90
    resign_consecutive: int = 5
    resign_min_move: int = 20
    resign_disabled_frac: float = 0.20  # 20% of games never resign
    resign_min_child_visits_frac: float = 0.05

    # Replay buffer
    buffer_size: int = 500_000

    # Training
    batch_size: int = 256
    train_steps_per_iter: int = 100
    lr_init: float = 0.01
    lr_final: float = 0.0001
    momentum: float = 0.9
    weight_decay: float = 1e-4

    # Checkpointing
    checkpoint_interval: int = 10  # iterations between checkpoints
    eval_interval: int = 10  # iterations between evaluation

    # Game
    komi: float = 7.5


# Preset configurations from PLAN.md
CONFIGS = {
    9: (
        ModelConfig(board_size=9, num_blocks=10, channels=128),
        TrainingConfig(
            num_simulations=400,
            dirichlet_alpha=0.11,
            num_games_per_iter=2048,
            # 500K raw positions = 2.64 GB. Same RAM as the original
            # 500K *augmented* config (which OOM'd because it stored
            # 62.5K raw × 8 aug at push time and pushed peak past 42 GB
            # via the savez transient). The buffer redesign moved
            # augmentation to sample time, so 500K slots now hold 500K
            # *distinct* positions — ~3 iters of effective history.
            buffer_size=500_000,
            max_game_moves=150,
            # Exploration phase covers ~half the avg game (~85 moves) so
            # mid-game positions stay diverse in the replay buffer.
            # temp_low=0.25 preserves runner-up move signal in training
            # targets instead of collapsing to one-hot argmax.
            temperature_moves=30,
            temperature_low=0.25,
            # Halved from default 0.01 starting at run 2 iter 22 to slow
            # the BN drift visible in the iter-9..21 weight audit. The
            # cosine schedule is otherwise unchanged.
            lr_init=0.005,
            checkpoint_interval=1,
            eval_interval=5,  # tighter than default so we catch drift early
        ),
    ),
    13: (
        ModelConfig(board_size=13, num_blocks=15, channels=128),
        TrainingConfig(
            num_simulations=600,
            dirichlet_alpha=0.07,  # ≈ 10 / avg_legal_moves (~150 for 13x13)
            num_games_per_iter=2048,
            buffer_size=1_000_000,
            max_game_moves=250,
            # 13x13 games average ~120 moves (vs ~85 on 9x9). Keep
            # the exploration window at ~1/3 of the avg game like the
            # tuned 9x9 run 2 preset — 40 moves here. temperature_low
            # 0.25 (vs default 0.1) preserves runner-up move signal in
            # the training targets instead of collapsing to argmax.
            temperature_moves=40,
            temperature_low=0.25,
        ),
    ),
    19: (
        ModelConfig(board_size=19, num_blocks=20, channels=256),
        TrainingConfig(
            num_simulations=800,
            dirichlet_alpha=0.03,
            num_games_per_iter=1024,
            buffer_size=1_000_000,
            max_game_moves=400,
            temperature_moves=30,
        ),
    ),
}
