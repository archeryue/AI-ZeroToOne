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
    # Pass-collapse floor — below this move number, the pass action is
    # stripped from the MCTS visit distribution before sampling. 0
    # disables the gate (9x9 preset). 13x13 run4c sets this to 60
    # because the derived-value architecture teaches the net to pass
    # on "settled territory" from iter 0/1 late-game data, and the
    # lesson leaks into early-game positions causing pass-pass game
    # ends at move ~40-60. Stored policy target is unmodified.
    pass_min_move: int = 0

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
    # Value-loss weight in total loss. Default 1.0 preserves Phase 1
    # 9x9 behavior; 13x13 preset raises it to counter value-head
    # cannibalization where policy_loss (~5) dominates value_loss
    # (~0.9), leaving the value head under-trained — see
    # PHASE_TWO_TRAINING.md Problem 4.
    value_loss_weight: float = 1.0
    # Ownership-loss weight (KataGo-style auxiliary head, added in
    # run4). Default 0.0 preserves the 9x9 recipe — the 9x9 net
    # converged without this and we don't want to perturb it. The
    # 13x13 preset turns it on. The supervision signal is per-cell
    # BCE-with-logits, so the loss magnitude is ~0.5-0.7 on cold data
    # (uniform predictions) and drops as the head learns. Weight 1.5
    # follows KataGo's published range.
    ownership_loss_weight: float = 0.0

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
            # Dropped from 600 for wall-time: at 600 sims/move on a
            # 15b×128ch net with 256 parallel games, iter time is ~58
            # min, which means 60 iters ≈ 58 hours. 400 sims brings
            # that to ~38 min/iter (~38 h total) for a mild MCTS
            # quality hit — and it's exactly the value AlphaGo Zero
            # used on 13x13, so it's the published default not a
            # shortcut. Bump back to 600 if the strength curve looks
            # anemic after a few iters.
            num_simulations=400,
            dirichlet_alpha=0.07,  # ≈ 10 / avg_legal_moves (~150 for 13x13)
            num_games_per_iter=2048,
            # run4b → run4c: dropped 256 → 128 to shrink peak RSS.
            # run4b iter 2 died silently from host-level SIGKILL on the
            # shared container (load avg 5-6, we were a top-3 RSS
            # tenant at ~30 GB). Halving parallel games halves the
            # MCTS tree state: ~19 GB → ~10 GB peak. Total RSS drops
            # to ~20 GB, moving us out of the OOM-killer top-candidate
            # range. GPU throughput is ~unchanged at 128 × vl_batch(8)
            # = 1024 (still saturates a 4090 for a 15b×128ch net per
            # HARDWARE_NOTES.md), so iter wall-time is essentially flat.
            num_parallel_games=128,
            buffer_size=1_000_000,
            max_game_moves=250,
            # 13x13 games average ~120 moves (vs ~85 on 9x9). Keep
            # the exploration window at ~1/3 of the avg game like the
            # tuned 9x9 run 2 preset — 40 moves here. temperature_low
            # 0.25 (vs default 0.1) preserves runner-up move signal in
            # the training targets instead of collapsing to argmax.
            temperature_moves=40,
            temperature_low=0.25,
            # Batch 1024: 4× bigger than default 256 for more stable
            # value head gradients on noisy ±1 labels. Applied at iter 3.
            batch_size=1024,
            # lr=0.00125: batch 256→1024 (4×) with linear scaling rule
            # (0.005/4). Applied at iter 3 after v_loss reversed at
            # iter 2. Bigger batch stabilizes value head gradients on
            # noisy ±1 labels.
            lr_init=0.00125,
            # Raised 40 → 80 and threshold -0.90 → -0.95 after run4b
            # iter 2 resumed and produced ~50 avg moves/game from an
            # iter-1-trained checkpoint (vs iter 0/1 at 173/182). The
            # ownership head, after 60 cumulative SGD steps, is
            # confident enough that tanh(0.02 · Σ(2σ(own)−1)) sometimes
            # crosses −0.9 at move 40+, triggering resigns that bias
            # the buffer toward short games. -0.95 requires much more
            # confident losing predictions; move-floor 80 ≈ 45 % of a
            # typical 180-move 13x13 game so resign can only fire past
            # the midgame. Credible-child cross-check stays on as the
            # second line of defense.
            resign_min_move=80,
            resign_threshold=-0.95,
            # Pass-collapse floor — run4c observed iter 2 avg moves
            # drop 182 → 69 because the iter-1-trained ownership head
            # made MCTS see most positions as "settled" and sample
            # pass, collapsing games via consecutive passes long
            # before the 80-move resign floor. Blocking pass before
            # move 60 forces games to play out the middlegame. 60
            # ≈ 40 % of a healthy 170-move game; after move 60 the
            # net is free to pass if territory is genuinely settled.
            pass_min_move=60,
            # Standard loss: policy + value + ownership auxiliary.
            # Reverted from run4's vlw=0 / derived-value experiment.
            # Value head is now a standard MLP (same as 9x9 / AlphaGo
            # Zero). Ownership head is a KataGo-style auxiliary that
            # regularizes the trunk with dense per-cell supervision.
            value_loss_weight=1.0,
            # 50 steps × batch 1024 = 51,200 samples/iter ≈ 29%
            # coverage of the ~175k positions from 1024 games/iter.
            train_steps_per_iter=50,
            # Ownership weight 1.5 — KataGo-range auxiliary supervision.
            # Dense per-cell labels regularize the trunk without
            # replacing the value head.
            ownership_loss_weight=1.5,
            # Eval every iter instead of every 5 iters for run2's first
            # few iters — we need iter-by-iter strength visibility to
            # confirm the fix actually works. Can revert to 5 once the
            # strength curve is confirmed monotonically improving.
            eval_interval=1,
            # Per-iter checkpoints are mandatory for Phase 2 post-hoc
            # Bradley-Terry tournaments — Phase 1 proved you cannot
            # reconstruct weight-drift trajectories from sparse
            # snapshots. 60 iters × ~36 MB ≈ 2.2 GB, trivial.
            checkpoint_interval=1,
            # (eval_interval=1 is set above for run2 visibility)
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
