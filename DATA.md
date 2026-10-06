# Data

No dataset files are distributed with this repository (the FLAME model and the
HAAR hairstyle assets are MPI-licensed, and BIWI is license-restricted).
Everything below is either obtainable from its original source or exactly
re-creatable with the packaged recipes.

## 1. Synthetic training corpus (25,000 FLAME renders)

250 identities (200 with HAAR hairstyles, 50 bald) x 2 HDRI environments x
10 expressions x 5 viewpoints, 54 skin textures, rendered in Blender 3.6.23
with [blended_in_flames](https://github.com/filby89/blended_in_flames)
(commit `fdfc5e1`).

Recipe (scripts in `data_generation/`):

| Step | Script | Notes |
|---|---|---|
| 20k hair corpus | `render_static_final_cuda.py` via `launch_script.sh` | exact arguments preserved: 200 ids (`identity_ids.list`) x 2 lighting x 10 expr x 5 views, CYCLES 64 samples, focal/distance 0.5-2.0, yaw +-80, pitch -40..30, seed 42 |
| 5k bald corpus | `render_static_multi_expr_cam.py` | 50 identities, no hair |
| Face boxes | `evaluation/precompute-face-bbox.py` | writes the per-view `face_bbox.npy` the training loader requires |

Required external assets (obtain under their own licenses): the FLAME 2023
model files (`flame2023_no_jaw.pkl`, `FLAME_masks.pkl`, landmark embeddings)
from [MPI FLAME](https://flame.is.tue.mpg.de/), the HAAR hairstyle assets from
the [HAAR](https://haar.is.tue.mpg.de/) authors (MPI license), 665 PolyHaven
HDRIs (CC0), and the 54 albedo textures. Rendering is not bit-exact across
Blender/GPU builds; it reproduces the corpus distribution.

Set `SYNTHETIC_HAIR_ROOT` / `SYNTHETIC_NOHAIR_ROOT` in `assets.env` to the two
rendered roots. Training pairs are sampled online by the loader (seeded); no
pair manifest is needed.

## 2. BIWI (evaluation)

Obtain the BIWI Kinect Head Pose Database from its authors (Fanelli et al.,
IJCV 2013; ETH Zurich license). `BIWI_ROOT` must point at the `faces_0`
layout:

```
faces_0/
  01/ rgb.cal  frame_*_rgb.png  frame_*_pose.txt ...
  01.obj        <- per-subject head mesh from the original release (required)
  ...
  24/  24.obj
```

Poses are converted from the depth to the RGB camera frame by the evaluator
using each subject's `rgb.cal`; detection is online MTCNN (facenet-pytorch,
weights auto-downloaded). The fixed hard/easy benchmark pair lists ship in
`protocols/` and must be used verbatim for Tables 2-3 comparisons.
