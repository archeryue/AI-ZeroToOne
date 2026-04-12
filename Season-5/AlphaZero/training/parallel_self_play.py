"""Multi-threaded parallel self-play with batched GPU inference.

Architecture (from PLAN.md):
  Orchestrator thread: GPU inference (BF16, batched), harvesting, training
  Worker threads 1-5: C++ SelfPlayWorker (GIL released for true parallelism)
  Each worker manages ~52 games, total 256 parallel games

Per tick:
  1. Workers: select 8 leaves/game → write observations to shared buffer
  2. Barrier sync
  3. Orchestrator: batch GPU forward pass on all observations
  4. Barrier sync
  5. Workers: process NN results → expand, backup, complete moves
"""

import os
import threading
import time
import numpy as np
import torch

import go_engine

from model.config import ModelConfig, TrainingConfig
from training.replay_buffer import ReplayBuffer

WORKER_CLASS = {
    9: go_engine.SelfPlayWorker9,
    13: go_engine.SelfPlayWorker13,
    19: go_engine.SelfPlayWorker19,
}


def _make_sp_config(train_cfg: TrainingConfig) -> go_engine.SelfPlayConfig:
    """Convert Python TrainingConfig to C++ SelfPlayConfig."""
    cfg = go_engine.SelfPlayConfig()
    cfg.komi = train_cfg.komi
    cfg.c_puct = train_cfg.c_puct
    cfg.dirichlet_alpha = train_cfg.dirichlet_alpha
    cfg.dirichlet_epsilon = train_cfg.dirichlet_epsilon
    cfg.vl_batch = train_cfg.virtual_loss_batch
    cfg.num_sims = train_cfg.num_simulations
    cfg.temp_moves = train_cfg.temperature_moves
    cfg.temp_high = train_cfg.temperature_high
    cfg.temp_low = train_cfg.temperature_low
    cfg.resign_threshold = train_cfg.resign_threshold
    cfg.resign_consecutive = train_cfg.resign_consecutive
    cfg.resign_disabled_frac = train_cfg.resign_disabled_frac
    cfg.max_game_moves = train_cfg.max_game_moves
    return cfg


class ParallelSelfPlay:
    """Multi-threaded self-play with batched GPU inference."""

    def __init__(self, net: torch.nn.Module, device: torch.device,
                 model_cfg: ModelConfig, train_cfg: TrainingConfig,
                 num_workers: int = 5):
        self.net = net
        self.device = device
        self.model_cfg = model_cfg
        self.train_cfg = train_cfg
        self.num_workers = num_workers
        self.use_cuda = device.type == "cuda"

        # Optional: torch.compile the inference net. Set AZ_COMPILE=0 to
        # skip (useful for smoke tests where warmup dominates).
        self.infer_net = net
        if self.use_cuda and os.environ.get("AZ_COMPILE", "1") != "0":
            try:
                self.infer_net = torch.compile(net, mode="max-autotune")
            except Exception as e:
                print(f"[ParallelSelfPlay] torch.compile disabled: {e}")
                self.infer_net = net

        N = model_cfg.board_size
        total_games = train_cfg.num_parallel_games
        games_per_worker = total_games // num_workers
        leftover = total_games - games_per_worker * num_workers

        vl_batch = train_cfg.virtual_loss_batch
        self.games_per_worker = []
        for i in range(num_workers):
            g = games_per_worker + (1 if i < leftover else 0)
            self.games_per_worker.append(g)

        max_nn_per_worker = [g * vl_batch for g in self.games_per_worker]
        total_max_nn = sum(max_nn_per_worker)
        self.total_max_nn = total_max_nn

        # Per-worker OWNED numpy arrays (not views into a shared buffer).
        # Passing sliced views into the C++ binding was correlated with a
        # SIGSEGV that always fired on the first tick_select of a fresh
        # run_games call. Owned per-worker buffers eliminate the shared-
        # base-array pointer/refcount path entirely.
        self.worker_obs = [
            np.zeros((mnn, 17, N, N), dtype=np.float32)
            for mnn in max_nn_per_worker
        ]
        self.worker_policy = [
            np.zeros((mnn, N * N + 1), dtype=np.float32)
            for mnn in max_nn_per_worker
        ]
        self.worker_value = [
            np.zeros(mnn, dtype=np.float32)
            for mnn in max_nn_per_worker
        ]

        # Separate pinned tensors for fast DMA. H2D: copy obs_buffer →
        # obs_pinned → GPU.  D2H: GPU → policy_pinned → policy_buffer.
        if self.use_cuda:
            self.obs_pinned = torch.empty(
                (total_max_nn, 17, N, N), dtype=torch.float32, pin_memory=True)
            self.policy_pinned = torch.empty(
                (total_max_nn, N * N + 1), dtype=torch.float32, pin_memory=True)
            self.value_pinned = torch.empty(
                total_max_nn, dtype=torch.float32, pin_memory=True)

        # Per-worker offsets into shared buffers
        self.worker_offsets = []
        offset = 0
        for mnn in max_nn_per_worker:
            self.worker_offsets.append(offset)
            offset += mnn
        self.max_nn_per_worker = max_nn_per_worker

        # Per-worker nn counts (written by workers, read by orchestrator)
        self.nn_counts = [0] * num_workers

        # Create C++ workers
        sp_cfg = _make_sp_config(train_cfg)
        WorkerClass = WORKER_CLASS[N]
        self.workers = []
        for i in range(num_workers):
            w = WorkerClass(self.games_per_worker[i], sp_cfg, seed=i * 10000)
            self.workers.append(w)

        # Synchronization
        self.select_barrier = threading.Barrier(num_workers + 1)
        self.process_barrier = threading.Barrier(num_workers + 1)
        self.stop_event = threading.Event()
        self.threads = []

    def _worker_thread(self, worker_id: int):
        """Worker thread loop: tick_select → barrier → tick_process → repeat."""
        worker = self.workers[worker_id]
        obs_buf = self.worker_obs[worker_id]
        pol_buf = self.worker_policy[worker_id]
        val_buf = self.worker_value[worker_id]

        while not self.stop_event.is_set():
            # Phase 1: Select leaves (GIL released inside C++)
            nn_count = worker.tick_select(obs_buf)
            self.nn_counts[worker_id] = nn_count

            try:
                self.select_barrier.wait()
                self.process_barrier.wait()
            except threading.BrokenBarrierError:
                break

            # Phase 2: Process results (GIL released inside C++)
            if nn_count > 0:
                worker.tick_process(pol_buf[:nn_count], val_buf[:nn_count])
            else:
                worker.tick_process(None, None)

            # Restart any finished games
            worker.restart_completed()

    def run_games(self, target_games: int, buffer: ReplayBuffer) -> dict:
        """Run parallel self-play until target_games are completed.

        Safe to call repeatedly on the same instance — barriers are reset,
        worker threads spawned fresh, and completed-game counts tracked as
        a delta from the entry point. Reusing one instance across training
        iterations avoids the per-iter SEGV we hit when constructing new
        pinned buffers and C++ workers every loop.
        """
        self.net.eval()
        self.stop_event.clear()
        # Barriers were `.abort()`d at the end of any prior call — reset
        # them so threads can actually use them again.
        self.select_barrier.reset()
        self.process_barrier.reset()
        N = self.model_cfg.board_size
        ACTIONS = N * N + 1

        # Track games completed DURING this call (workers keep a cumulative
        # counter across calls; we need a delta).
        games_done_start = sum(w.games_done for w in self.workers)

        # Start worker threads
        self.threads = []
        for i in range(self.num_workers):
            t = threading.Thread(target=self._worker_thread, args=(i,),
                                 daemon=True)
            t.start()
            self.threads.append(t)

        total_games = 0
        total_positions = 0
        black_wins = 0
        ticks = 0
        t_start = time.time()

        try:
            while total_games < target_games:
                # Wait for all workers to finish selecting
                self.select_barrier.wait()

                # Harvest completed data (safe: workers are at barrier)
                for worker in self.workers:
                    obs_np, pol_np, val_np, count = worker.harvest()
                    if count > 0:
                        for j in range(count):
                            buffer.push(obs_np[j], pol_np[j], val_np[j],
                                        augment=True)
                        total_positions += count

                total_games = (sum(w.games_done for w in self.workers)
                               - games_done_start)

                # GPU inference on the full fixed-size batch. Each worker
                # now owns its own contiguous obs array; we stage each one
                # into the correct slice of obs_pinned, then DMA to the GPU
                # in a single transfer. The fixed total shape lets
                # torch.compile lock onto a single kernel shape. Gap slots
                # (where nn_count < max_nn) contain stale data but their
                # outputs are harmless — each worker only reads back the
                # first nn_count entries of its own policy/value buffer.
                total_nn = sum(self.nn_counts)
                if total_nn > 0:
                    if self.use_cuda:
                        for i, obs_buf in enumerate(self.worker_obs):
                            off = self.worker_offsets[i]
                            mx = self.max_nn_per_worker[i]
                            self.obs_pinned[off:off + mx].copy_(
                                torch.from_numpy(obs_buf))
                        obs_tensor = self.obs_pinned.to(
                            self.device, non_blocking=True)
                    else:
                        # CPU path: stitch owned buffers into one big tensor
                        obs_cat = np.concatenate(
                            [ob for ob in self.worker_obs], axis=0)
                        obs_tensor = torch.from_numpy(obs_cat)

                    with torch.no_grad():
                        if self.use_cuda:
                            with torch.amp.autocast("cuda"):
                                logits, values = self.infer_net(obs_tensor)
                        else:
                            logits, values = self.infer_net(obs_tensor)
                        policies = torch.softmax(logits, dim=-1)

                    if self.use_cuda:
                        # D2H into pinned; sync; then split by worker offset
                        # into each worker's owned policy/value arrays.
                        self.policy_pinned.copy_(
                            policies, non_blocking=True)
                        self.value_pinned.copy_(
                            values, non_blocking=True)
                        torch.cuda.synchronize()
                        pol_np = self.policy_pinned.numpy()
                        val_np = self.value_pinned.numpy()
                        for i in range(self.num_workers):
                            off = self.worker_offsets[i]
                            mx = self.max_nn_per_worker[i]
                            np.copyto(self.worker_policy[i],
                                      pol_np[off:off + mx])
                            np.copyto(self.worker_value[i],
                                      val_np[off:off + mx])
                    else:
                        pol_np = policies.numpy()
                        val_np = values.numpy()
                        for i in range(self.num_workers):
                            off = self.worker_offsets[i]
                            mx = self.max_nn_per_worker[i]
                            np.copyto(self.worker_policy[i],
                                      pol_np[off:off + mx])
                            np.copyto(self.worker_value[i],
                                      val_np[off:off + mx])

                # Signal workers to process results
                self.process_barrier.wait()
                ticks += 1

        finally:
            # Stop workers
            self.stop_event.set()
            self.select_barrier.abort()
            self.process_barrier.abort()
            for t in self.threads:
                t.join(timeout=5.0)

        # Final harvest
        for worker in self.workers:
            obs_np, pol_np, val_np, count = worker.harvest()
            if count > 0:
                for j in range(count):
                    buffer.push(obs_np[j], pol_np[j], val_np[j], augment=True)
                total_positions += count

        total_games = (sum(w.games_done for w in self.workers)
                       - games_done_start)
        elapsed = time.time() - t_start

        return {
            "games": total_games,
            "positions": total_positions,
            "positions_augmented": total_positions * 8,
            "ticks": ticks,
            "time": elapsed,
            "games_per_sec": total_games / max(elapsed, 0.001),
            "ticks_per_sec": ticks / max(elapsed, 0.001),
            "buffer_size": len(buffer),
        }
