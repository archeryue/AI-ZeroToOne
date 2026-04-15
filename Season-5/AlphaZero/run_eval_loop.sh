#!/usr/bin/env bash
# Background evaluator: watches for new checkpoint_NNNN.pt in ckpt_dir
# and runs eval-vs-random on each in a fresh Python process.
#
# Usage: run_eval_loop.sh <ckpt_dir> <results_file>
#
# Results written one-line-per-iter to $results_file:
#   iter=5 wr=0.340 games=50 sec=119.4 ts=12:34:56
#
# Each eval runs in a subprocess with a 300s timeout to catch the
# handover's "eval hangs mid-forward" issue. Fresh process per
# checkpoint avoids the state-accumulation hang observed when 3+
# checkpoints were loaded in one process.

set -u

CKPT_DIR="${1:-checkpoints/13x13_run4b}"
RESULTS="${2:-logs/eval_vs_random.log}"
NUM_GAMES="${NUM_GAMES:-50}"
POLL_INTERVAL="${POLL_INTERVAL:-30}"

cd "$(dirname "$0")"
mkdir -p "$(dirname "$RESULTS")"
touch "$RESULTS"

echo "[eval-loop] starting. ckpt_dir=$CKPT_DIR results=$RESULTS games=$NUM_GAMES"

while true; do
    for ckpt in "$CKPT_DIR"/checkpoint_*.pt; do
        [[ -f "$ckpt" ]] || continue
        iter=$(basename "$ckpt" .pt | sed 's/checkpoint_0*//')
        [[ -z "$iter" ]] && iter=0
        if grep -q "^iter=$iter " "$RESULTS" 2>/dev/null; then
            continue  # already evaluated
        fi
        echo "[eval-loop] evaluating iter $iter from $ckpt"
        t0=$(date +%s)
        out=$(PYTHONPATH=engine AZ_COMPILE=0 timeout 300 python -c "
import os, sys, time
sys.path.insert(0, '.')
for v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(v,'1')
import torch
torch.set_num_threads(1)
try: torch.set_num_interop_threads(1)
except RuntimeError: pass
torch.set_float32_matmul_precision('high')
from model.config import CONFIGS
from model.network import AlphaZeroNet
from training.trainer import Trainer
from training.train import evaluate_vs_random
device = torch.device('cuda')
mcfg, tcfg = CONFIGS[13]
net = AlphaZeroNet(mcfg).to(device)
trainer = Trainer(net, tcfg, device)
trainer.load_checkpoint('$ckpt')
wr = evaluate_vs_random(net, device, mcfg, tcfg, num_games=$NUM_GAMES)
print(f'WR {wr:.4f}')
" 2>&1)
        rc=$?
        t1=$(date +%s)
        dt=$((t1 - t0))
        wr=$(echo "$out" | grep -oE "^WR [0-9.]+" | awk '{print $2}')
        ts=$(date +%H:%M:%S)
        if [[ -n "$wr" && $rc -eq 0 ]]; then
            echo "iter=$iter wr=$wr games=$NUM_GAMES sec=$dt ts=$ts" >> "$RESULTS"
            echo "[eval-loop] iter $iter wr=$wr ($dt s)"
        else
            echo "iter=$iter wr=FAILED rc=$rc sec=$dt ts=$ts" >> "$RESULTS"
            echo "[eval-loop] iter $iter FAILED (rc=$rc, ${dt}s)"
        fi
    done
    sleep "$POLL_INTERVAL"
done
