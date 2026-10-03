#!/usr/bin/env bash
# Launch one training run of the learning-curve experiment.
#
#   scripts/launch_run.sh <run-name> <train-split> [extra train.py flags...]
#
#   scripts/launch_run.sh q-w25  train_w25  --negative-queue 1000
#   scripts/launch_run.sh qa-w25 train_w25  --negative-queue 1000 --augment
#   scripts/launch_run.sh w100   train_w100 --negative-queue 1000 --augment
#
# Everything that must be identical across runs lives in this file rather than
# in your shell. A shell function and its variables do not survive a new tmux
# pane, a reconnect, or `tmux kill-server`, and when they vanish the run does
# not fail loudly — it starts with defaults on the wrong data root
# (timeline.md, box B blocker 1, and again 2026-09-29).
#
# To reproduce the pre-queue lc-w25 configuration, pass --negative-queue 0.

set -euo pipefail

if [ $# -lt 2 ]; then
    echo "usage: $0 <run-name> <train-split> [extra train.py flags...]" >&2
    exit 2
fi
RUN=$1
SPLIT=$2
shift 2

: "${DATA:?not set. export DATA=/opt/dlami/nvme/data, add it to ~/.bashrc, and run 'tmux kill-server' if tmux started before the export}"

REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"

for required in \
    "$DATA/manifests/$SPLIT/pairs.jsonl" \
    "$DATA/manifests/$SPLIT/tracks.jsonl" \
    "$DATA/manifests/val50/pairs.jsonl" \
    "$DATA/manifests/train_w100/pairs.jsonl" \
    "$REPO/phase1_layer_weights.pt"
do
    if [ ! -f "$required" ]; then
        echo "missing: $required" >&2
        echo "splits present under $DATA/manifests:" >&2
        ls -1 "$DATA/manifests" 2>/dev/null | sed 's/^/  /' >&2 || echo "  (no manifests directory)" >&2
        exit 1
    fi
done

# One budget for every run in both stages, derived from the *full* pair set so
# the number does not depend on which split this run uses: 10 passes over
# train_w100 at 4 pairs/step. lc-w25 used 87372; if this prints anything else,
# the manifests were rebuilt and nothing is comparable to it any more.
STEPS=$(( $(wc -l < "$DATA/manifests/train_w100/pairs.jsonl") * 10 / 4 ))

mkdir -p "checkpoints/$RUN" "runs/$RUN"
echo "run=$RUN  split=$SPLIT  steps=$STEPS  data=$DATA  extra=$*"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -u scripts/train.py \
    --data-root "$DATA" \
    --train-split "$SPLIT" \
    --val-split val50 \
    --checkpoint-dir "checkpoints/$RUN" --log-dir "runs/$RUN" \
    --layer-weights phase1_layer_weights.pt --freeze-layer-weights \
    --batch-pairs 4 --batch-tracks 4 --num-workers 4 \
    --max-steps "$STEPS" --val-every 2500 --checkpoint-every 500 \
    --select-metric val/content_work_map --seed 0 \
    "$@" 2>&1 | tee -a "checkpoints/$RUN/train.log"
