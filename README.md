# VGGT-HPE: Reframing Head Pose Estimation as Relative Pose Prediction

Official implementation of **VGGT-HPE** (CVPR Workshops 2026).

VGGT-HPE casts head pose estimation as *relative* rigid-pose prediction
between an anchor image with known pose and a query image, built on the
[VGGT](https://github.com/facebookresearch/vggt) geometry foundation model
with LoRA fine-tuning on purely synthetic FLAME renderings. Despite zero real
training images, it reaches state-of-the-art rotation accuracy on BIWI.

[Paper (CVF Open Access)](https://openaccess.thecvf.com/) · [Project page](https://vasilikivas.github.io/VGGT-HPE)

## Results (BIWI, shared MTCNN protocol)

| Method | Yaw | Pitch | Roll | MAE | Train data |
|---|---|---|---|---|---|
| 6DRepNet | 3.74 | 4.95 | 3.04 | 3.91 | mixed |
| TokenHPE-v1 | 5.57 | 6.23 | 3.79 | 5.20 | mixed |
| TRG | 4.58 | 7.18 | 3.68 | 5.15 | mixed |
| VGGT-HPE-Abs (ours) | 4.90 | 7.01 | 3.53 | 5.15 | synthetic |
| **VGGT-HPE (Rel., ours)** | **2.24** | **3.04** | 3.17 | **2.82** | synthetic |

## Repository layout

```
vggt/            VGGT model code with the head-pose adaptations (training interface)
training/        trainer, losses, data loader, Hydra configs (training/config/)
evaluation/      BIWI + synthetic evaluators; vggt_compat/ pins the evaluator's
                 model interface (selected automatically by run_biwi_eval.py)
scripts/         train.sh, eval_biwi.sh, eval_synthetic.sh
protocols/       fixed 360-pair hard/easy BIWI benchmarks (Tables 2-3)
data_generation/ synthetic-corpus creation recipe (see DATA.md)
```

## Installation

```bash
conda env create -f environment.yml   # python 3.10, torch 2.2.2 + cu121
conda activate vggt-hpe
```

Then configure asset paths:

```bash
cp assets.env.example assets.env      # edit the paths inside
```

See [DATA.md](DATA.md) for datasets and [CHECKPOINTS.md](CHECKPOINTS.md) for
weights. A quick environment check (no data needed):

```bash
python evaluation/run_biwi_eval.py --self-test
```

## Evaluation (inference)

Full BIWI (Table 1):

```bash
OUTPUT_DIR=outputs/biwi_main scripts/eval_biwi.sh
```

Fixed hard / easy benchmarks (Tables 2-3; do **not** regenerate the pair CSVs
when comparing against the paper):

```bash
PAIR_CSV=protocols/table2_hard_pairs.csv OUTPUT_DIR=outputs/biwi_hard scripts/eval_biwi.sh
PAIR_CSV=protocols/table3_easy_pairs.csv OUTPUT_DIR=outputs/biwi_easy scripts/eval_biwi.sh
```

Results are written to `OUTPUT_DIR/biwi_eval_results.csv`; the paper's
VGGT-HPE (Rel.) numbers are the `B_no_flip` variant row. Synthetic validation:
`scripts/eval_synthetic.sh`.

## Training

```bash
OUTPUT_DIR=outputs/train scripts/train.sh                 # main model (flame_h2c_lora_pose_f1_only_500)
CONFIG=flame_h2c_abs_single_lora scripts/train.sh         # absolute baseline
```

All Table-4 ablation configs are in `training/config/`. Training runs on a
single A100-64GB-class GPU; pairs are sampled online from the synthetic
corpus (see DATA.md), and per-view `face_bbox.npy` files must be precomputed
with `evaluation/precompute-face-bbox.py` first.

## Citation

```bibtex
@inproceedings{vasileiou2026vggthpe,
  title     = {VGGT-HPE: Reframing Head Pose Estimation as Relative Pose Prediction},
  author    = {Vasileiou, Vasiliki and Filntisis, Panagiotis P. and Maragos, Petros and Daniilidis, Kostas},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)},
  year      = {2026}
}
```

## License & acknowledgements

Built on [VGGT](https://github.com/facebookresearch/vggt) (research license,
retained as `LICENSE.txt`) and [FLAME](https://flame.is.tue.mpg.de/). Training
data is rendered with [blended_in_flames](https://github.com/filby89/blended_in_flames).
We thank the EuroHPC Joint Undertaking for access to Leonardo (CINECA).
