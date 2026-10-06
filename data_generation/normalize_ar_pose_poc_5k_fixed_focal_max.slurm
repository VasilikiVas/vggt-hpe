#!/bin/bash
#SBATCH --job-name=fix_focal_ar5k
#SBATCH --output=/leonardo_work/EUHPC_D32_089/head_pose/blended_in_flames/logs/fix_focal_ar5k_%j.out
#SBATCH --error=/leonardo_work/EUHPC_D32_089/head_pose/blended_in_flames/logs/fix_focal_ar5k_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --partition=boost_usr_prod
#SBATCH --account=EUHPC_D32_089

set -euo pipefail

ROOT="/leonardo_work/EUHPC_D32_089/head_pose/blended_in_flames"
BLENDER="${ROOT}/blender-3.6.23-linux-x64/blender"
SCRIPT="${ROOT}/normalize_dataset_focal.py"
SRC="${ROOT}/sample_outputs/ar_pose_poc_5k"
DST="${ROOT}/sample_outputs/ar_pose_poc_5k_fixed_focal_max"

mkdir -p "${ROOT}/logs"

echo "=== Fixed Focal Copy Job ==="
echo "Job ID: ${SLURM_JOB_ID:-manual}"
echo "Node: ${SLURMD_NODENAME:-local}"
echo "Start: $(date)"
echo "Source: ${SRC}"
echo "Destination: ${DST}"
echo "Target mode: max"
echo "============================"

cd "${ROOT}"

"${BLENDER}" --background \
  --python "${SCRIPT}" -- \
  --src "${SRC}" \
  --dst "${DST}" \
  --target_mode max \
  --overwrite 1

echo "=== Job Complete ==="
echo "End: $(date)"
