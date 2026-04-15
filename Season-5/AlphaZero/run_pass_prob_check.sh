#!/usr/bin/env bash
# Background checker: for every checkpoint_NNNN.pt, record the
# pass-action logit + softmax probability on an empty 13x13 board.
# Writes one-line-per-iter to logs/pass_prob.log.
#
# Serves as an early-warning signal for the pass-collapse regression:
# if pass_prob climbs past ~5 % we know the v2 target-fix is
# insufficient and we need to intervene before the policy breaks.

set -u
CKPT_DIR="${1:-checkpoints/13x13_run4b}"
RESULTS="${2:-logs/pass_prob.log}"
POLL_INTERVAL="${POLL_INTERVAL:-30}"

cd "$(dirname "$0")"
mkdir -p "$(dirname "$RESULTS")"
touch "$RESULTS"

echo "[pass-prob-loop] starting. ckpt_dir=$CKPT_DIR results=$RESULTS"

while true; do
    for ckpt in "$CKPT_DIR"/checkpoint_*.pt; do
        [[ -f "$ckpt" ]] || continue
        iter=$(basename "$ckpt" .pt | sed 's/checkpoint_0*//')
        [[ -z "$iter" ]] && iter=0
        if grep -q "^iter=$iter " "$RESULTS" 2>/dev/null; then
            continue
        fi
        out=$(PYTHONPATH=engine python -c "
import torch
from model.config import CONFIGS
from model.network import AlphaZeroNet
m, t = CONFIGS[13]
net = AlphaZeroNet(m)
state = torch.load('$ckpt', map_location='cpu', weights_only=False)
net.load_state_dict(state['model_state_dict'])
net.eval()
obs = torch.zeros(1, 17, 13, 13)
obs[0, 16, :, :] = 1
with torch.no_grad():
    pi, v, own = net(obs)
import torch.nn.functional as F
p = F.softmax(pi[0], dim=0)
argmax = pi.argmax().item()
pass_logit = pi[0, 169].item()
max_logit = pi.max().item()
pass_prob = p[169].item()
print(f'argmax={argmax} pass_logit={pass_logit:+.4f} max_logit={max_logit:+.4f} pass_prob={pass_prob:.6f}')
" 2>/dev/null)
        ts=$(date +%H:%M:%S)
        if [[ -n "$out" ]]; then
            echo "iter=$iter $out ts=$ts" >> "$RESULTS"
            echo "[pass-prob-loop] iter $iter $out"
        fi
    done
    sleep "$POLL_INTERVAL"
done
