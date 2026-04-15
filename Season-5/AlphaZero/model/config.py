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
            # Restored to 256 after raising MAX_TREE_NODES 200k → 1M.
            # With the bigger cap per tree uses ~75 MB × 256 = ~19 GB
            # for MCTS state, which still fits under the 35 GB budget
            # alongside buffer + savez transient + model/compile
            # (~30 GB total, ~5 GB headroom). The bigger GPU batch
            # (256 × vl_batch(8) = 2048 vs 1024 at 128 parallel)
            # amortizes per-tick CPU+sync overhead and gives ~20-40%
            # faster iters.
            num_parallel_games=256,
            buffer_size=1_000_000,
            max_game_moves=250,
            # 13x13 games average ~120 moves (vs ~85 on 9x9). Keep
            # the exploration window at ~1/3 of the avg game like the
            # tuned 9x9 run 2 preset — 40 moves here. temperature_low
            # 0.25 (vs default 0.1) preserves runner-up move signal in
            # the training targets instead of collapsing to argmax.
            temperature_moves=40,
            temperature_low=0.25,
            # Halved from the 0.01 default to match the 9x9 run 2
            # fix for BN drift. Run 1 used 0.01 and exhibited the
            # iter-4→19 regression; run 2 at 0.005 was stable. Cheap
            # insurance on a bigger net (15b vs 10b) and a bigger game.
            lr_init=0.005,
            # Raised from the 20 default after run1 iter 1 saw games
            # collapse to 41 avg moves (vs iter 0's 144) and value loss
            # rise 0.83 → 0.90 — classic early-resign data bias. On a
            # 13x13 game with max_game_moves=250, a move floor of 40 is
            # ~27% of a typical 150-move game, matching the proportion
            # the 9x9 run 2 fix used (20/85 ≈ 23%). Credible-child
            # cross-check stays on as the second line of defense.
            resign_min_move=40,
            # value_loss_weight=0.0 for run4 — this is NOT the same
            # "turn off the value head" as old runs; it only disables
            # the direct value-loss gradient. Value is now DERIVED
            # from ownership (see model/network.py AlphaZeroNet): the
            # value head has only two learnable scalars (value_scale,
            # value_bias) and its output is tanh(k·Σ(2σ(own_logits)−1)+b).
            #
            # The offline A/B (training/_phase2_run4_offline_ab.py)
            # showed that with derived value, any nonzero vlw hijacks
            # the ownership head into predicting per-cell values that
            # sum to noisy game outcomes — directly conflicting with
            # the ownership loss's "predict real territory" target.
            # With vlw=0, ownership loss alone trains the ownership
            # head on dense per-cell real-game labels, and derived
            # value reads off that. A6 recipe achieved held-out
            # v_mse=0.9631 — the ONLY recipe across run1/2/3/4 offline
            # A/Bs that went below the cold floor of ~1.00.
            value_loss_weight=0.0,
            # train_steps_per_iter=30 to prevent per-iter overfit.
            # 30*60=1800 SGD steps total; with ownership's 169x
            # supervision density, effective label count is ~300k
            # per-cell targets across the run.
            train_steps_per_iter=30,
            # Ownership weight 2.0 — the primary (and effectively
            # only, with vlw=0) supervision signal for the trunk
            # and the derived-value readout. Offline A/B winner.
            ownership_loss_weight=2.0,
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
