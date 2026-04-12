"""Multi-threaded reproducer for the tick_select SIGSEGV.

Same threading layout as ParallelSelfPlay: 5 worker threads calling
`tick_select` in parallel, synchronized via a select/process Barrier
pair, with the main thread acting as the "orchestrator". This bypasses
GPU inference (fakes the NN output) so it won't compete with a live
training run.

Runs for N seconds and exits 0 if no segfault.
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
          "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(v, "1")

import sys
import time
import argparse
import threading
import faulthandler
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
faulthandler.enable()

import numpy as np
import torch
torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

import go_engine
from model.config import CONFIGS
from training.parallel_self_play import _make_sp_config

parser = argparse.ArgumentParser()
parser.add_argument("seconds", type=int, default=60, nargs="?")
args = parser.parse_args()

N = 9
model_cfg, train_cfg = CONFIGS[N]
sp_cfg = _make_sp_config(train_cfg)

num_workers = 5
games_per_worker = train_cfg.num_parallel_games // num_workers
leftover = train_cfg.num_parallel_games - games_per_worker * num_workers
games_per_worker_list = [
    games_per_worker + (1 if i < leftover else 0) for i in range(num_workers)
]

vl_batch = train_cfg.virtual_loss_batch
max_nn_per_worker = [g * vl_batch for g in games_per_worker_list]
total_max_nn = sum(max_nn_per_worker)
worker_offsets = []
off = 0
for m in max_nn_per_worker:
    worker_offsets.append(off)
    off += m

obs_buffer = np.zeros((total_max_nn, 17, N, N), dtype=np.float32)
policy_buffer = np.zeros((total_max_nn, N * N + 1), dtype=np.float32)
value_buffer = np.zeros(total_max_nn, dtype=np.float32)

# Fill policy/value with dummy NN outputs so workers never see NaNs.
policy_buffer.fill(1.0 / (N * N + 1))
value_buffer.fill(0.0)

workers = [
    go_engine.SelfPlayWorker9(games_per_worker_list[i], sp_cfg, seed=i * 777)
    for i in range(num_workers)
]

select_barrier = threading.Barrier(num_workers + 1)
process_barrier = threading.Barrier(num_workers + 1)
stop_event = threading.Event()
nn_counts = [0] * num_workers

def _worker_thread(wid):
    worker = workers[wid]
    off = worker_offsets[wid]
    mx = max_nn_per_worker[wid]
    obs_slice = obs_buffer[off:off + mx]
    pol_slice = policy_buffer[off:off + mx]
    val_slice = value_buffer[off:off + mx]
    while not stop_event.is_set():
        nn_count = worker.tick_select(obs_slice)
        nn_counts[wid] = nn_count
        try:
            select_barrier.wait()
            process_barrier.wait()
        except threading.BrokenBarrierError:
            break
        if nn_count > 0:
            worker.tick_process(pol_slice[:nn_count], val_slice[:nn_count])
        else:
            worker.tick_process(None, None)
        worker.restart_completed()

threads = []
for i in range(num_workers):
    t = threading.Thread(target=_worker_thread, args=(i,), daemon=True)
    t.start()
    threads.append(t)

print(f"Multi-threaded repro: {num_workers} workers, {total_max_nn} obs/tick")
print(f"Running for {args.seconds}s...")

t0 = time.perf_counter()
ticks = 0
try:
    while time.perf_counter() - t0 < args.seconds:
        select_barrier.wait()
        # (skipping GPU — policy/value buffers already have fake NN outputs)
        process_barrier.wait()
        ticks += 1
        if ticks % 100 == 0:
            print(f"  {ticks} ticks, "
                  f"{ticks / (time.perf_counter() - t0):.0f}/s, "
                  f"games_done={sum(w.games_done for w in workers)}")
finally:
    stop_event.set()
    select_barrier.abort()
    process_barrier.abort()
    for t in threads:
        t.join(timeout=5.0)

elapsed = time.perf_counter() - t0
print(f"OK: {ticks} ticks in {elapsed:.1f}s — no segfault")
print(f"Games completed: {sum(w.games_done for w in workers)}")
