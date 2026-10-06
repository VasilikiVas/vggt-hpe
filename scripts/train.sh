#!/usr/bin/env bash
# Train VGGT-HPE (LoRA on the VGGT camera branch, synthetic FLAME pairs).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ASSET_ENV="${ASSET_ENV:-${ROOT}/assets.env}"
[[ ! -f "${ASSET_ENV}" ]] || source "${ASSET_ENV}"

: "${VGGT_BASE_CHECKPOINT:?Set VGGT_BASE_CHECKPOINT or provide assets.env}"
: "${SYNTHETIC_HAIR_ROOT:?Set SYNTHETIC_HAIR_ROOT or provide assets.env}"
: "${SYNTHETIC_NOHAIR_ROOT:?Set SYNTHETIC_NOHAIR_ROOT or provide assets.env}"

OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/outputs/training}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MASTER_PORT="${MASTER_PORT:-29500}"
CONFIG="${CONFIG:-flame_h2c_lora_pose_f1_only_500}"
mkdir -p "${OUTPUT_DIR}"

export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${OUTPUT_DIR}/wandb"
export TMPDIR="${OUTPUT_DIR}/runtime"
mkdir -p "${WANDB_DIR}" "${TMPDIR}"

INIT_CHECKPOINT="${RESUME_CHECKPOINT:-${VGGT_BASE_CHECKPOINT}}"
cd "${ROOT}/training"
exec torchrun --master_port "${MASTER_PORT}" --nproc_per_node "${NPROC_PER_NODE}" launch.py \
  --config "${CONFIG}" \
  "checkpoint.resume_checkpoint_path=${INIT_CHECKPOINT}" \
  "checkpoint.save_dir=${OUTPUT_DIR}/ckpts" \
  "logging.log_dir=${OUTPUT_DIR}" \
  "data.train.dataset.dataset_configs.0.root_dirs.0.path=${SYNTHETIC_HAIR_ROOT}" \
  "data.train.dataset.dataset_configs.0.root_dirs.1.path=${SYNTHETIC_NOHAIR_ROOT}" \
  "data.val.dataset.dataset_configs.0.root_dirs.0.path=${SYNTHETIC_HAIR_ROOT}" \
  "data.val.dataset.dataset_configs.0.root_dirs.1.path=${SYNTHETIC_NOHAIR_ROOT}" \
  "$@"
