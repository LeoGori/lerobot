#!/usr/bin/env bash
# Usage: ./train.sh [extra lerobot-train flags]
#
# Runs in the foreground: the training output is shown live in your terminal (or in the tmux session you run
# it from) and is also appended to $OUTPUT_ROOT/logs/train_<timestamp>.log. To survive closing the terminal /
# SSH, start it from tmux:
#   tmux new -s groot_train
#   ./train.sh                       # detach with Ctrl-b d, reattach with `tmux attach -t groot_train`

# Must be executed, not sourced: `set -e` and `exit` below would otherwise apply to (and close) your shell.
if [ "${BASH_SOURCE[0]}" != "$0" ]; then
  echo "Run this as ./train.sh (or bash train.sh), not '. train.sh' / 'source train.sh'." >&2
  return 1
fi
set -euo pipefail

# Run from the repo root so the relative --dataset.root path resolves.
SCRIPT="$(readlink -f "${BASH_SOURCE[0]}")"
cd "$(dirname "$SCRIPT")"

REPO_ID="TAPTIC/graspglasshaptictattileno_reduced_GR00T17"

# No checkpoints are written during training (--save_checkpoint=false): the only artifact is the final model,
# pushed to the Hub when training ends. `push_to_hub` stages the ~13 GB model (FP32) in $TMPDIR before uploading,
# so point OUTPUT_ROOT at a disk with space if the root disk is tight: `OUTPUT_ROOT=/path ./train.sh`.
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/train}"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
# A fresh dir per run: lerobot-train refuses to start if --output_dir already exists.
OUTPUT_DIR="$OUTPUT_ROOT/graspglasshaptictattile_reduced_GR00T17_$RUN_ID"
LOG_FILE="$OUTPUT_ROOT/logs/train_$RUN_ID.log"
MIN_FREE_GB="${MIN_FREE_GB:-16}"  # ~13 GB staging area for the final push + headroom.

# Prefer `uv run` (project convention); fall back to lerobot-train on PATH.
if command -v uv >/dev/null 2>&1; then
  RUN=(uv run)
else
  RUN=()
fi

mkdir -p "$OUTPUT_ROOT/tmp" "$OUTPUT_ROOT/logs"
export TMPDIR="$(readlink -f "$OUTPUT_ROOT/tmp")"

free_gb=$(df --output=avail -BG "$OUTPUT_ROOT" | tail -1 | tr -dc '0-9')
if [ "$free_gb" -lt "$MIN_FREE_GB" ]; then
  echo "Only ${free_gb} GB free under $OUTPUT_ROOT (need ~${MIN_FREE_GB} GB to stage the final model push)." >&2
  echo "Free some space, set OUTPUT_ROOT to a disk with more, or lower MIN_FREE_GB to override." >&2
  exit 1
fi

# The model is only uploaded at the very end, so fail now (not after 20k steps) if the login is missing.
"${RUN[@]}" python - <<'EOF'
import sys

from huggingface_hub import HfApi

try:
    auth = HfApi().whoami()["auth"]
except Exception as exc:  # noqa: BLE001
    sys.exit(f"Not logged in to Hugging Face ({exc}). Run `hf auth login` with a write token.")
# OAuth logins carry no role; access tokens do.
if auth.get("accessToken", {}).get("role") == "read":
    sys.exit("The Hugging Face token is read-only; log in with a write token (`hf auth login`).")
EOF

# Loss curves go to Weights & Biases. Without a login, log offline (./wandb) so the run doesn't crash:
# upload it later with `wandb sync wandb/offline-run-*`.
if [ -z "${WANDB_API_KEY:-}" ] && ! grep -qs "api.wandb.ai" ~/.netrc; then
  echo "Not logged in to wandb (run \`wandb login\`): logging offline." >&2
  export WANDB_MODE=offline
fi

# Show everything on the terminal and append it to the log file (tee -i: Ctrl-C stops training, not the logging).
exec > >(tee -ia "$LOG_FILE") 2>&1
echo "[$(date '+%F %T')] run $RUN_ID | host $(hostname) | output_dir $OUTPUT_DIR | log $LOG_FILE"

# Training loss / grad norm / lr are logged every --log_freq steps. --dataset.eval_split holds out 10% of the
# episodes from training, drawn at random (--dataset.eval_split_shuffle, reproducible from --seed; the chosen
# episodes are printed at startup); --eval_steps computes the loss on them (see --eval_steps below).
set +e
"${RUN[@]}" lerobot-train \
  --dataset.repo_id=TAPTIC/graspglasstattileno_reduced \
  --dataset.root=data/graspglasshaptictattileno_reduced \
  --dataset.image_transforms.enable=true \
  --dataset.eval_split=0.1 \
  --dataset.eval_split_shuffle=true \
  --policy.type=groot \
  --policy.device=cuda \
  --policy.base_model_path=nvidia/GR00T-N1.7-3B \
  --policy.embodiment_tag=new_embodiment \
  --policy.use_bf16=true \
  --policy.push_to_hub=true \
  --policy.repo_id="$REPO_ID" \
  --seed=42 \
  --batch_size=8 \
  --steps=20000 \
  --save_checkpoint=true \
  --save_freq=1000 \
  --use_policy_training_preset=true \
  --env_eval_freq=0 \
  --eval_steps=1000 \
  --log_freq=50 \
  --output_dir="$OUTPUT_DIR" \
  --job_name=graspglasshaptictattileno_reduced_groot \
  --wandb.enable=true \
  --policy.private=true \
  --wandb.disable_artifact=true \
  "$@"
status=$?
echo "[$(date '+%F %T')] lerobot-train exited with status $status"
sleep 1  # let the tee process flush the last lines before the shell prompt returns
exit "$status"
