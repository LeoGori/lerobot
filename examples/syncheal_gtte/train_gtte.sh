#!/usr/bin/env bash
# Fine-tune GR00T N1.7 on a SyncHeal dataset with frozen FT + gaze temporal encoders (GTTE).
#
# Usage (from the repo root, after building the dataset with build_gtte_dataset.py):
#   ./examples/syncheal_gtte/train_gtte.sh [extra lerobot-train flags]
#
# Environment overrides:
#   FT_ENCODER / GAZE_ENCODER   encoder export folders (tactile-encoder-ergocub, syncheal/{ft,gaze}-ae branches)
#   DATASET_ROOT / DATASET_ID   GTTE dataset built by build_gtte_dataset.py
#   STEPS / BATCH_SIZE          training length and batch size
#   HUB_REPO_ID                 if set, push the final policy there (private) instead of keeping a local copy
#   OUTPUT_ROOT                 where runs and logs go (default outputs/train)
#
# Loss curves go to Trackio's local database: `trackio show --project syncheal_gtte`
# (add --trackio.space_id=<user>/<space> to log to a Hugging Face Space instead).
#
# Runs in the foreground; start it from tmux to survive closing the terminal:
#   tmux new -s gtte && ./examples/syncheal_gtte/train_gtte.sh

if [ "${BASH_SOURCE[0]}" != "$0" ]; then
  echo "Run this as ./train_gtte.sh (or bash train_gtte.sh), not with 'source'." >&2
  return 1
fi
set -euo pipefail

SCRIPT="$(readlink -f "${BASH_SOURCE[0]}")"
cd "$(dirname "$SCRIPT")/../.."  # repo root, so the relative paths below resolve

ENCODERS_REPO="${ENCODERS_REPO:-../tactile-encoder-ergocub}"
FT_ENCODER="$(readlink -f "${FT_ENCODER:-$ENCODERS_REPO/outputs/syncheal_ft_ae/export}")"
GAZE_ENCODER="$(readlink -f "${GAZE_ENCODER:-$ENCODERS_REPO/outputs/syncheal_gaze_ae/export}")"
DATASET_ID="${DATASET_ID:-SyncHeal/move-handle-test-gtte}"
DATASET_ROOT="${DATASET_ROOT:-data/move-handle-test-gtte}"
STEPS="${STEPS:-40000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
HUB_REPO_ID="${HUB_REPO_ID:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/train}"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="$OUTPUT_ROOT/move_handle_gtte_groot17_$RUN_ID"
LOG_FILE="$OUTPUT_ROOT/logs/train_gtte_$RUN_ID.log"

for path in "$FT_ENCODER" "$GAZE_ENCODER"; do
  if [ ! -f "$path/artifacts/model_state.pt" ]; then
    echo "Missing encoder export: $path (train it with 'python -m syncheal_ae.train --config configs/...')." >&2
    exit 1
  fi
done
if [ ! -f "$DATASET_ROOT/meta/info.json" ]; then
  echo "Missing GTTE dataset at $DATASET_ROOT. Build it first:" >&2
  echo "  python examples/syncheal_gtte/build_gtte_dataset.py --encoder $FT_ENCODER --encoder $GAZE_ENCODER --out-root $DATASET_ROOT" >&2
  exit 1
fi

if command -v uv >/dev/null 2>&1; then RUN=(uv run); else RUN=(); fi

mkdir -p "$OUTPUT_ROOT/tmp" "$OUTPUT_ROOT/logs"
export TMPDIR="$(readlink -f "$OUTPUT_ROOT/tmp")"

# Without a Hub repo the final checkpoint (~13 GB, FP32) is saved locally; with one it is only pushed.
if [ -n "$HUB_REPO_ID" ]; then
  SAVE_FLAGS=(--save_checkpoint=false --policy.push_to_hub=true --policy.repo_id="$HUB_REPO_ID" --policy.private=true)
else
  SAVE_FLAGS=(--save_checkpoint=true --save_freq="$STEPS" --policy.push_to_hub=false)
fi

exec > >(tee -ia "$LOG_FILE") 2>&1
echo "[$(date '+%F %T')] run $RUN_ID | output_dir $OUTPUT_DIR | ft $FT_ENCODER | gaze $GAZE_ENCODER"

# GTTE: the frozen encoders turn 2 s of history, observation.ft_window (200, 6) and observation.gaze_window (60, 9:
# gaze + torso/neck joints), into 32 + 8 standardized latents, written into the zero-padded GR00T state slots after the 46-dim
# observation.state (see src/lerobot/policies/groot/temporal_encoders.py).
# chunk_size=16 / n_action_steps=8 follow the TTE paper (1.6 s chunk at 10 Hz, re-plan after 8 actions).
# eval_split holds out 10% of the episodes with seed 42: the same episodes the encoders used for validation.
set +e
"${RUN[@]}" lerobot-train \
  --dataset.repo_id="$DATASET_ID" \
  --dataset.root="$DATASET_ROOT" \
  --dataset.image_transforms.enable=true \
  --dataset.eval_split=0.1 \
  --dataset.eval_split_shuffle=true \
  --policy.type=groot \
  --policy.device=cuda \
  --policy.base_model_path=nvidia/GR00T-N1.7-3B \
  --policy.embodiment_tag=new_embodiment \
  --policy.temporal_encoders="{\"observation.ft_window\": \"$FT_ENCODER\", \"observation.gaze_window\": \"$GAZE_ENCODER\"}" \
  --policy.chunk_size=16 \
  --policy.n_action_steps=8 \
  --policy.use_bf16=true \
  --seed=42 \
  --batch_size="$BATCH_SIZE" \
  --steps="$STEPS" \
  --use_policy_training_preset=true \
  --env_eval_freq=0 \
  --eval_steps=4000 \
  --log_freq=200 \
  --output_dir="$OUTPUT_DIR" \
  --job_name=move_handle_gtte_groot17 \
  --trackio.enable=true \
  --trackio.project=syncheal_gtte \
  "${SAVE_FLAGS[@]}" \
  "$@"
status=$?
echo "[$(date '+%F %T')] lerobot-train exited with status $status"
sleep 1
exit "$status"
