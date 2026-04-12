"""Scan GPU forward cost vs batch size and compile, to pick the sweet spot."""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
torch.set_num_threads(1); torch.set_num_interop_threads(1)

from model.config import CONFIGS
from model.network import AlphaZeroNet

N = 9
model_cfg, train_cfg = CONFIGS[N]
device = torch.device("cuda")

net = AlphaZeroNet(model_cfg).to(device).eval()

def bench(mod, bs, n=50):
    obs = torch.randn(bs, 17, N, N, device=device)
    with torch.no_grad(), torch.amp.autocast("cuda"):
        for _ in range(5): mod(obs)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad(), torch.amp.autocast("cuda"):
        for _ in range(n): mod(obs)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1000

print(f"{'batch':>6}  {'time(ms)':>10}  {'obs/ms':>10}  {'games/s*':>10}")
print("-" * 50)
# *games/s = (batch/vl)/(tick_time/1000)/(2500/parallel)
# simplification: parallel = batch/vl, so games/s = parallel * 1000 / (tick * 2500)
# tick ~ forward+9ms overhead
VL = 8
for bs in [64, 128, 256, 512, 1024, 2048, 4096, 8192]:
    dt = bench(net, bs)
    obs_per_ms = bs / dt
    parallel = bs // VL
    tick = dt + 9  # estimated total tick ~= gpu + 9ms overhead
    games_s = parallel * 1000 / (tick * 2500)
    print(f"{bs:>6}  {dt:>10.2f}  {obs_per_ms:>10.1f}  {games_s:>10.2f}")

print()
print("--- Same scan with torch.compile ---")
try:
    cnet = torch.compile(net, mode="reduce-overhead")
    for bs in [256, 512, 1024, 2048, 4096]:
        dt = bench(cnet, bs, n=30)
        parallel = bs // VL
        tick = dt + 9
        games_s = parallel * 1000 / (tick * 2500)
        print(f"{bs:>6}  {dt:>10.2f}  games/s* = {games_s:>6.2f}")
except Exception as e:
    print(f"torch.compile failed: {e}")
