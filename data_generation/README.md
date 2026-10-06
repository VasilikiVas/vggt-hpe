# Synthetic training-data generation (FLAME + Blender)

The 25,000-image training corpus (20k with hair + 5k bald) was rendered with
the `blended_in_flames` pipeline. This directory packages the recipe scripts;
the renderer framework, Blender, and the licensed/large assets stay external.

## Provenance

- Renderer framework: `blended_in_flames`
  (https://github.com/filby89/blended_in_flames, commit `fdfc5e1`). The
  paper-specific scripts below were **untracked** files in two checkouts:
  - `/leonardo_work/EUHPC_D32_089/head_pose/blended_in_flames` (this project's
    checkout; holds the rendered corpora under `sample_outputs/`), and
  - `/leonardo_work/EUHPC_D32_089/blended_in_flames_2` (collaborator checkout;
    the actual 20k-hair render recipe, files dated 2026-03-07).
- Blender 3.6.23 (bundled inside the `blended_in_flames` checkout).

## Scripts in this directory

| File | Role |
|---|---|
| `launch_script.sh` | Exact 20k-hair render arguments: 200 identities x 2 lighting setups x 10 expressions x 5 views, CYCLES 64 samples, focal and distance ranges 0.5-2.0, yaw +-80 deg, pitch -40..30 deg, seed 42. |
| `render_static_final_cuda.py` | The 20k-hair renderer entry point (HTCondor/CUDA variant actually used). |
| `identity_ids.list` | The 200 rendered identity IDs. |
| `render_static_multi_expr_cam.py` | **Likely** the 5k no-hair (bald) renderer: it is the only script with lighting setups and no hair, and its mtime matches the January 2026 render. No launcher or argument record for the 5k corpus survives, so this attribution is unverified. |
| `normalize_dataset_focal.py` | Fixed-focal conversion (target fx ~= 2630.8; unchanged files hardlinked; writes `focal_normalization_manifest.json` per copy). Used by ReVA-HPE, not by the CVPRW VGGT-HPE training. |
| `normalize_dataset_20k_hair_v2_fixed_focal_max.slurm`, `normalize_ar_pose_poc_5k_fixed_focal_max.slurm` | The 2026-05 fixed-focal conversion launchers. |
| `check_fixed_focal_overlay.py` | Optional validation of the fixed-focal copies. |

## External assets NOT copied here

From `blended_in_flames/assets/` (licensed or large):
`flame2023_no_jaw.pkl`, `FLAME_masks.pkl`, `landmark_embedding.npy`,
`head_prior.obj`, `env_maps/` (665 PolyHaven HDRIs), `face_textures/`
(54 albedo textures), plus `flame_render.blend` / `flame_render_w_hair.blend`
and `src/FLAME`. The HAAR hairstyle point clouds (~11 GB; **MPI-licensed — obtain via the HAAR
authors, not redistributable**) exist locally only in
`blended_in_flames_2/inference_results/infer_haar_*`.

## Rendered corpora (historical locations)

- `blended_in_flames/sample_outputs/dataset_20k_hair_v2` (~38 GB, 200 ids)
- `blended_in_flames/sample_outputs/ar_pose_poc_5k` (~15 GB, 50 ids)
- `*_fixed_focal_max` siblings (ReVA-HPE training inputs)

After rendering, per-view face boxes must be precomputed with
`evaluation/precompute-face-bbox.py` (writes `face_bbox.npy` next to each
view) before the training loader can run.

## Data-distribution policy (decided 2026-10)

**No rendered or derived data files will be distributed.** The HAAR hairstyle
assets are MPI-licensed (obtain from the HAAR authors under the MPI license),
FLAME is MPI-licensed, and the evaluation datasets are license-restricted, so
the published reproduction path is: exact creation recipes (this directory),
exact arguments and seeds, and SHA-256 hashes of every derived artifact so an
independently regenerated or privately shared copy can be authenticated.
Rendering is not guaranteed bit-exact across Blender/GPU builds; the hashes
authenticate the originals, the recipes reproduce the distribution.
