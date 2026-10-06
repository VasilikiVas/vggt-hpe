# Checkpoints

## Ours (released by the authors)

| Checkpoint | Size | SHA-256 | Download |
|---|---:|---|---|
| VGGT-HPE `checkpoint_305.pt` (main relative model) | 6,307,141,652 B | `d01f8f494778f0432aa90ca65c3efbc6ae2cd3d2a326bcf2bc6e6cd857bf4537` | `TO_BE_PUBLISHED` |
| VGGT-HPE-Abs `checkpoint.pt` (absolute baseline) | 5.87 GB | `d71d1be16b8333605b395bef55cda861b4686b782457cac4e87d76dfadaf73b9` | `TO_BE_PUBLISHED` |

Both contain VGGT-1B-derived weights and are released under the VGGT research
(non-commercial) license (`LICENSE.txt`). Verify downloads against the SHA-256
values above. Point `VGGT_HPE_CHECKPOINT` in `assets.env` at the main model.

## Third-party (download from the official sources)

| Asset | Source | SHA-256 |
|---|---|---|
| VGGT-1B `model.pt` (frozen base) | [facebookresearch/vggt](https://github.com/facebookresearch/vggt) official release | `d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0` |
| MTCNN weights | bundled with `facenet-pytorch==2.6.0` (auto-download) | — |

Set `VGGT_BASE_CHECKPOINT` to the VGGT-1B file. The optional baseline
comparisons in the evaluator (6DRepNet, TokenHPE-v1, TRG) require those
projects' official repositories and weights:
[6DRepNet](https://github.com/thohemp/6DRepNet) @`464b2ba`,
[TokenHPE](https://github.com/zc2023/TokenHPE) @`8651d36`,
[TRG-Release](https://github.com/asw91666/TRG-Release) @`3916650`.
