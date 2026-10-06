#!/usr/bin/env bash
# Evaluate a checkpoint on the held-out synthetic FLAME validation split.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ASSET_ENV="${ASSET_ENV:-${ROOT}/assets.env}"
[[ ! -f "${ASSET_ENV}" ]] || source "${ASSET_ENV}"
: "${VGGT_BASE_CHECKPOINT:?}"; : "${VGGT_HPE_CHECKPOINT:?}"
: "${SYNTHETIC_HAIR_ROOT:?}"; : "${SYNTHETIC_NOHAIR_ROOT:?}"
export PYTHONPATH="${ROOT}:${ROOT}/evaluation:${PYTHONPATH:-}"
exec "${PYTHON:-python}" "${ROOT}/evaluation/eval_flame_synthetic.py" \
  --base_checkpoint "${VGGT_BASE_CHECKPOINT}" \
  --lora_checkpoint "${VGGT_HPE_CHECKPOINT}" \
  --root_dirs "${SYNTHETIC_HAIR_ROOT}" "${SYNTHETIC_NOHAIR_ROOT}" \
  "$@"
