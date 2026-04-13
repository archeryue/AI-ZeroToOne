"""Micro-benchmark: where is self-play time going?

Breaks a real Phase 1 tick into its components so we can see
whether CPU workers, GPU inference, or Python orchestrator dominate.
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
          "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(v, "1")

import sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

import go_engine
from model.config import CONFIGS
from model.network import AlphaZeroNet
from training.parallel_self_play import ParallelSelfPlay, _make_sp_config
from training.replay_buffer import ReplayBuffer

N = 9
model_cfg, train_cfg = CONFIGS[N]
device = torch.device("cuda")

net = AlphaZeroNet(model_cfg).to(device).eval()
print(f"Model: {net.param_count():,} params")

# --- 1. GPU forward alone ---
B = train_cfg.num_parallel_games * train_cfg.virtual_loss_batch  # 2048
obs = torch.randn(B, 17, N, N, device=device, dtype=torch.float32)
with torch.no_grad(), torch.amp.autocast("cuda"):
    net(obs)  # warmup
torch.cuda.synchronize()

n_fwd = 50
t0 = time.perf_counter()
with torch.no_grad(), torch.amp.autocast("cuda"):
    for _ in range(n_fwd):
        net(obs)
torch.cuda.synchronize()
dt = (time.perf_counter() - t0) / n_fwd * 1000
print(f"[1] GPU forward (bs={B}, bf16): {dt:.2f} ms/call")

# --- 2. C++ workers doing tick_select alone ---
sp_cfg = _make_sp_config(train_cfg)
num_workers = 5
games_per_worker = train_cfg.num_parallel_games // num_workers
vl = train_cfg.virtual_loss_batch
workers = [go_engine.SelfPlayWorker9(games_per_worker, sp_cfg, seed=i*777)
           for i in range(num_workers)]
buf = np.zeros((games_per_worker * vl, 17, N, N), dtype=np.float32)

# Warmup
for w in workers:
    w.tick_select(buf)

import threading
n_rounds = 20
t0 = time.perf_counter()
for _ in range(n_rounds):
    threads = []
    results = [0] * num_workers
    def run(i):
        results[i] = workers[i].tick_select(buf)
    for i in range(num_workers):
        t = threading.Thread(target=run, args=(i,))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
dt_ws = (time.perf_counter() - t0) / n_rounds * 1000
print(f"[2] 5 workers tick_select in parallel: {dt_ws:.2f} ms/round "
      f"(~{sum(results)} obs/round)")

# --- 3. Full orchestrator loop (2 tick measurement) ---
sp = ParallelSelfPlay(net, device, model_cfg, train_cfg, num_workers=5)
buffer = ReplayBuffer(train_cfg.buffer_size, N)

# Run a small target so we can time the loop
print("\n[3] Running real orchestrator for ~10 s wall to measure ticks/sec...")
class LimitedSP(ParallelSelfPlay):
    def run_games_timed(self, seconds, buffer, warmup_s=10.0):
        self.net.eval()
        self.stop_event.clear()
        self.threads = []
        for i in range(self.num_workers):
            t = threading.Thread(target=self._worker_thread, args=(i,), daemon=True)
            t.start()
            self.threads.append(t)
        ticks = 0
        total_positions = 0
        t_start = time.perf_counter()
        tick_times = []
        ss_start = None  # steady-state window start
        ss_tick_times = []
        ss_games_start = 0
        ss_positions_start = 0
        try:
            while time.perf_counter() - t_start < seconds:
                tick_start = time.perf_counter()
                self.select_barrier.wait()
                # harvest
                harvest_start = time.perf_counter()
                for worker in self.workers:
                    obs_np, pol_np, val_np, count = worker.harvest()
                    if count > 0:
                        for j in range(count):
                            buffer.push(obs_np[j], pol_np[j], val_np[j])
                        total_positions += count
                harvest_dt = time.perf_counter() - harvest_start

                total_nn = sum(self.nn_counts)
                gpu_start = time.perf_counter()
                if total_nn > 0:
                    if self.use_cuda:
                        obs_tensor = self.obs_pinned.to(self.device, non_blocking=True)
                    with torch.no_grad(), torch.amp.autocast("cuda"):
                        logits, values = self.infer_net(obs_tensor)
                        policies = torch.softmax(logits, dim=-1)
                    if self.use_cuda:
                        self.policy_pinned.copy_(policies, non_blocking=True)
                        self.value_pinned.copy_(values, non_blocking=True)
                        torch.cuda.synchronize()
                gpu_dt = time.perf_counter() - gpu_start
                self.process_barrier.wait()
                tick_dt = time.perf_counter() - tick_start
                tick_times.append((tick_dt, harvest_dt, gpu_dt, total_nn))
                ticks += 1
                # Mark steady-state start once warmup elapses
                elapsed_now = time.perf_counter() - t_start
                if ss_start is None and elapsed_now >= warmup_s:
                    ss_start = time.perf_counter()
                    ss_games_start = sum(w.games_done for w in self.workers)
                    ss_positions_start = total_positions
                if ss_start is not None:
                    ss_tick_times.append((tick_dt, harvest_dt, gpu_dt, total_nn))
        finally:
            self.stop_event.set()
            self.select_barrier.abort()
            self.process_barrier.abort()
            for t in self.threads:
                t.join(timeout=5.0)

        elapsed = time.perf_counter() - t_start
        games_done = sum(w.games_done for w in self.workers)
        print(f"  Ticks: {ticks} in {elapsed:.1f}s → {ticks/elapsed:.1f} ticks/s")
        print(f"  Games completed: {games_done}")
        print(f"  Positions harvested: {total_positions}")

        def _report(label, tt):
            if not tt: return
            tot = np.array([x[0] for x in tt]) * 1000
            har = np.array([x[1] for x in tt]) * 1000
            gpu = np.array([x[2] for x in tt]) * 1000
            nn_counts = np.array([x[3] for x in tt])
            other = tot - har - gpu
            print(f"  [{label}] ({len(tt)} ticks)")
            print(f"    Tick total: mean={tot.mean():.2f} ms  p50={np.median(tot):.2f}  p90={np.percentile(tot,90):.2f}")
            print(f"    GPU+xfer:   mean={gpu.mean():.2f} ms  p50={np.median(gpu):.2f}  p90={np.percentile(gpu,90):.2f}")
            print(f"    Other:      mean={other.mean():.2f} ms  p50={np.median(other):.2f}")
        _report("ALL", tick_times)
        _report("STEADY", ss_tick_times)

        if ss_start is not None:
            ss_elapsed = time.perf_counter() - ss_start
            ss_games = games_done - ss_games_start
            ss_games_s = ss_games / max(ss_elapsed, 1e-6)
            print(f"  [STEADY window] {ss_elapsed:.1f}s → {ss_games} games → "
                  f"{ss_games_s:.2f} games/s")

sp2 = LimitedSP(net, device, model_cfg, train_cfg, num_workers=5)
sp2.run_games_timed(120, buffer, warmup_s=60.0)
