#!/usr/bin/env bash
set -euo pipefail

# Condor-friendly single-job launcher:
# runs exactly one identity (1 identity per job).

cd /fast/pfilntisis/sila/blended_in_flames_2

PY="/lustre/fast/fast/pfilntisis/.virtualenvs/TEMPEH/bin/python"
SITE="$($PY -c 'import site; print(site.getsitepackages()[0])')"
BLENDER="/lustre/home/pfilntisis/my_blender/blender-3.6.5-linux-x64/blender"
if [ ! -x "$BLENDER" ]; then
    echo "Error: BLENDER binary not found or not executable: $BLENDER"
    exit 1
fi

TOTAL_IDENTITIES=200
NUM_IDENTITIES_PER_JOB=1

OUT_DIR="/fast/pfilntisis/sila/blended_in_flames_2/sample_outputs/dataset_20k_hair_v2"
HAIR_ROOT="/fast/pfilntisis/sila/blended_in_flames_2/inference_results"
mkdir -p "$OUT_DIR"

IDENTITY_ID="${1:-}"
if [ -z "$IDENTITY_ID" ]; then
    echo "Usage: $0 <identity_id>"
    exit 1
fi
if ! [[ "$IDENTITY_ID" =~ ^[0-9]+$ ]]; then
    echo "Error: identity_id must be a non-negative integer, got: $IDENTITY_ID"
    exit 1
fi
if [ "$IDENTITY_ID" -ge "$TOTAL_IDENTITIES" ]; then
    echo "Error: identity_id ($IDENTITY_ID) must be < TOTAL_IDENTITIES ($TOTAL_IDENTITIES)"
    exit 1
fi

if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    GPU_BIND="${CUDA_VISIBLE_DEVICES}"
elif [ -n "${_CONDOR_AssignedGPUs:-}" ]; then
    GPU_BIND="${_CONDOR_AssignedGPUs%%,*}"
else
    GPU_BIND="0"
fi

COMMON_ARGS="\
    --mesh_file FLAME \
    --hair_file_male ${HAIR_ROOT}/infer_haar_man \
    --hair_file_female ${HAIR_ROOT}/infer_haar_woman \
    --female_prob 0.5 \
    --out_dir ${OUT_DIR} \
    --render_engine CYCLES \
    --background_visible 1 \
    --randomize_focal_length_range 0.5,2.0 \
    --texture_path /fast/pfilntisis/sila/blended_in_flames_2/assets/face_textures/merged_textures \
    --envmap_path /fast/pfilntisis/sila/blended_in_flames_2/assets/env_maps/PolyHaven/HDRI/1k/ \
    --num_identities ${NUM_IDENTITIES_PER_JOB} \
    --num_lighting_setups 2 \
    --num_expressions 10 \
    --views_per_expression 5 \
    --randomize_env_map_rotation_z 0,360 \
    --save_output_mesh obj \
    --randomize_left_right_angle -80,80 \
    --randomize_up_down_angle -40,30 \
    --randomize_camera_tilt -30,30 \
    --randomize_camera_distance 0.5,2.0 \
    --randomize_lookat_offset_x -0.5,0.5 \
    --randomize_lookat_offset_y -0.05,0.05 \
    --randomize_lookat_offset_z -0.05,0.05 \
    --render_samples 64 \
    --hair_width 0.00007 \
    --hair_color -1 \
    --seed 42"

echo "═══════════════════════════════════════════════════"
echo "  Condor single-identity render"
echo "  GPU binding: ${GPU_BIND}"
echo "  Identity: ${IDENTITY_ID}"
echo "  Identities per job: ${NUM_IDENTITIES_PER_JOB}"
echo "  Output: ${OUT_DIR}"
echo "═══════════════════════════════════════════════════"
echo ""

START="${IDENTITY_ID}"
END=$((IDENTITY_ID + 1))

echo "Identity range (end-exclusive): ${START}-${END}"

CUDA_VISIBLE_DEVICES="${GPU_BIND}" \
EGL_PLATFORM=surfaceless \
PYTHONPATH="${SITE}:${PYTHONPATH:-}" \
"$BLENDER" flame_render_w_hair.blend \
    --python render_static_final_cuda.py --background -- \
    ${COMMON_ARGS} \
    --start_index "${START}" --end_index "${END}"
