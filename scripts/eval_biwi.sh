#!/usr/bin/env bash
# Evaluate VGGT-HPE (relative) on BIWI. Set PAIR_CSV to protocols/table2_hard_pairs.csv
# or protocols/table3_easy_pairs.csv for the hard/easy benchmarks (default: full BIWI).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ASSET_ENV="${ASSET_ENV:-${ROOT}/assets.env}"
[[ ! -f "${ASSET_ENV}" ]] || source "${ASSET_ENV}"

: "${VGGT_BASE_CHECKPOINT:?Set VGGT_BASE_CHECKPOINT (VGGT-1B model.pt) or provide assets.env}"
: "${VGGT_HPE_CHECKPOINT:?Set VGGT_HPE_CHECKPOINT (checkpoint_305.pt) or provide assets.env}"
: "${BIWI_ROOT:?Set BIWI_ROOT (BIWI faces_0 directory) or provide assets.env}"

PYTHON="${PYTHON:-python}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/outputs/biwi}"
GPU="${GPU:-0}"
BATCH_SIZE="${BATCH_SIZE:-64}"
mkdir -p "${OUTPUT_DIR}" "${OUTPUT_DIR}/runtime"

export PYTHONPATH="${ROOT}/evaluation/vggt_compat:${ROOT}:${ROOT}/evaluation:${PYTHONPATH:-}"
export TMPDIR="${OUTPUT_DIR}/runtime"
export MPLCONFIGDIR="${OUTPUT_DIR}/runtime/matplotlib"
export TORCH_HOME="${TORCH_CACHE:-${OUTPUT_DIR}/runtime/torch}"
mkdir -p "${MPLCONFIGDIR}" "${TORCH_HOME}"

extra=()
[[ -z "${PAIR_CSV:-}" ]] || extra+=(--pair_csv "${PAIR_CSV}")

exec "${PYTHON}" "${ROOT}/evaluation/run_biwi_eval.py" \
  --biwi_dir "${BIWI_ROOT}" \
  --base_checkpoint "${VGGT_BASE_CHECKPOINT}" \
  --lora_checkpoint "${VGGT_HPE_CHECKPOINT}" \
  --vggt_pose_mode relative_h2c \
  --face_detector mtcnn --mtcnn_ad 0.4 --crop_size 256 \
  --batch_size "${BATCH_SIZE}" --gpu "${GPU}" \
  --no_sixdrepnet --no_tokenhpe --no_whenet --no_trg --no_sixdof_face \
  --vis_dir "${OUTPUT_DIR}" \
  "${extra[@]}" "$@"
