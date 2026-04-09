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

    # Resign
    resign_threshold: float = -0.95
    resign_consecutive: int = 3
    resign_disabled_frac: float = 0.1  # 10% of games never resign

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
            buffer_size=500_000,
            max_game_moves=150,
            temperature_moves=15,
        ),
    ),
    13: (
        ModelConfig(board_size=13, num_blocks=15, channels=128),
        TrainingConfig(
            num_simulations=600,
            dirichlet_alpha=0.07,
            num_games_per_iter=2048,
            buffer_size=1_000_000,
            max_game_moves=250,
            temperature_moves=20,
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
