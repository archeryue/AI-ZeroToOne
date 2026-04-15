#!/usr/bin/env bash
# Resilient training wrapper: relaunches train.py on non-zero exit.
#
# Background: run4b iter 2+ self-play dies silently after ~5 min on
# this shared host — most likely an external kernel-OOM SIGKILL under
# global memory pressure. The parent process can't prevent that, but
# it can retry with the latest checkpoint. Combined with intra-iter
# buffer persistence (see train.py --save-callback), each restart
# preserves the partial self-play harvest from the previous attempt.
#
# Usage:
#   run_resilient.sh <output_dir> [--iterations N] [extra train.py args...]
#
# Example:
#   ./run_resilient.sh checkpoints/13x13_run4c --iterations 60
#
# The script picks the highest-numbered checkpoint_XXXX.pt in
# output_dir and passes it as --checkpoint. If none exists, it starts
# fresh. Caps retries at MAX_RETRIES and aborts if the process dies in
# under MIN_RUNTIME_S (would be a deterministic crash loop).

set -u

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <output_dir> [train.py args...]" >&2
    exit 2
fi

OUTPUT_DIR="$1"
shift

MAX_RETRIES="${MAX_RETRIES:-30}"
MIN_RUNTIME_S="${MIN_RUNTIME_S:-60}"
COOLDOWN_S="${COOLDOWN_S:-5}"

mkdir -p "$OUTPUT_DIR"
LOG_DIR="$(dirname "$OUTPUT_DIR")/logs"
mkdir -p "$LOG_DIR" 2>/dev/null || true

cd "$(dirname "$0")"

attempt=0
while true; do
    attempt=$((attempt + 1))
    if [[ $attempt -gt $MAX_RETRIES ]]; then
        echo "[watchdog] exceeded MAX_RETRIES=$MAX_RETRIES, giving up" >&2
        exit 1
    fi

    latest_ckpt=""
    if compgen -G "$OUTPUT_DIR/checkpoint_*.pt" > /dev/null; then
        latest_ckpt="$(ls -1 "$OUTPUT_DIR"/checkpoint_*.pt | sort | tail -1)"
    fi

    ckpt_args=()
    if [[ -n "$latest_ckpt" ]]; then
        ckpt_args=(--checkpoint "$latest_ckpt")
        echo "[watchdog] attempt $attempt — resuming from $latest_ckpt" >&2
    else
        echo "[watchdog] attempt $attempt — starting fresh (no checkpoint)" >&2
    fi

    start_t=$(date +%s)
    set +e
    PYTHONUNBUFFERED=1 PYTHONPATH=engine python -m training.train \
        --output-dir "$OUTPUT_DIR" \
        "${ckpt_args[@]}" \
        "$@"
    rc=$?
    set -e
    end_t=$(date +%s)
    elapsed=$((end_t - start_t))

    echo "[watchdog] attempt $attempt exited rc=$rc after ${elapsed}s" >&2

    if [[ $rc -eq 0 ]]; then
        echo "[watchdog] training completed cleanly" >&2
        exit 0
    fi

    # Treat Ctrl-C / SIGTERM (130, 143) as user-requested stops.
    if [[ $rc -eq 130 || $rc -eq 143 ]]; then
        echo "[watchdog] user-interrupted (rc=$rc), not retrying" >&2
        exit $rc
    fi

    if [[ $elapsed -lt $MIN_RUNTIME_S ]]; then
        echo "[watchdog] process died in ${elapsed}s (< MIN_RUNTIME_S=$MIN_RUNTIME_S) — looks like a deterministic crash, aborting" >&2
        exit $rc
    fi

    echo "[watchdog] cooldown ${COOLDOWN_S}s then retry..." >&2
    sleep "$COOLDOWN_S"
done
