<div align="center">

# VGGT-HPE

### Reframing Head Pose Estimation as Relative Pose Prediction

<p>
  <a href="https://vasilikivas.github.io/">Vasiliki Vasileiou</a> ·
  <a href="https://filby89.github.io/">Panagiotis P. Filntisis</a> ·
  <a href="https://robotics.ntua.gr/members/maragos/">Petros Maragos</a> ·
  <a href="https://www.cis.upenn.edu/~kostas/">Kostas Daniilidis</a>
</p>

<p>
  <a href="https://vasilikivas.github.io/VGGT-HPE/">
    <img src="https://img.shields.io/badge/Project%20Page-Website-00CC00?style=for-the-badge" alt="Project Page">
  </a>
  <a href="https://vasilikivas.github.io/assets/pdf/2026127020.pdf">
    <img src="https://img.shields.io/badge/Paper-PDF-4F6D7A?style=for-the-badge" alt="Paper PDF">
  </a>
  <a href="https://github.com/VasilikiVas/vggt-hpe">
    <img src="https://img.shields.io/github/stars/VasilikiVas/vggt-hpe?style=for-the-badge&logo=github&label=Stars" alt="GitHub stars">
  </a>
  <img src="https://img.shields.io/badge/Code-Coming%20Soon-C86B49?style=for-the-badge" alt="Code coming soon">
</p>

</div>

<p align="center">
  <img src="assets/teaser.png" alt="VGGT-HPE teaser" width="100%">
</p>

## Overview

VGGT-HPE reframes monocular head pose estimation as **relative pose prediction** instead of direct absolute regression. Rather than forcing a model to internalize a dataset-specific canonical reference frame, we estimate the rigid transformation between a **query image** and an **anchor image with known pose**. Fine-tuned only on synthetic facial renderings, VGGT-HPE achieves state-of-the-art BIWI performance and shows that relative prediction becomes increasingly advantageous as pose difficulty grows.

> This repository is currently a polished placeholder while the codebase is being cleaned up for release. If this project is relevant to your work, star the repository to follow the public release.

## Why This Formulation Matters

- It removes the need to infer a hidden canonical pose from a single image.
- It turns head pose estimation into a geometric displacement problem between two visible head states.
- It allows the anchor to be chosen at test time, which makes prediction difficulty controllable.

## Method At A Glance

<p align="center">
  <img src="assets/method.png" alt="VGGT-HPE method overview" width="100%">
</p>

VGGT-HPE takes an **anchor-query pair**, uses the **VGGT camera branch** as backbone, and predicts the relative transformation from the anchor pose to the query pose. The model is adapted with parameter-efficient fine-tuning and trained exclusively on synthetic FLAME renderings with supervision on rotation, translation, and field of view.

## Qualitative Results

<p align="center">
  <img src="assets/results.png" alt="VGGT-HPE qualitative results on BIWI" width="100%">
</p>

Each row shows a different subject. From left to right: the query frame, the anchor frame with its known pose overlay, the ground-truth pose, our prediction, and several strong baselines.

## Planned Release

- Training and evaluation code
- Configuration files and reproducibility scripts
- Pretrained checkpoints
- Documentation for benchmark evaluation and usage

## Citation

If you use this work, please cite:

```bibtex
@inproceedings{vasileiou2026vggthpe,
  title={VGGT-HPE: Reframing Head Pose Estimation as Relative Pose Prediction},
  author={Vasileiou, Vasiliki and Filntisis, Panagiotis P. and Maragos, Petros and Daniilidis, Kostas},
  booktitle={CVPR Workshop},
  year={2026}
}
```

## Links

- Project page: https://vasilikivas.github.io/VGGT-HPE/
- Paper PDF: https://vasilikivas.github.io/assets/pdf/2026127020.pdf

