#!/usr/bin/env python3
"""Launcher for the BIWI evaluator (`eval_biwi_crop_mtcnn_square_compare.py`).

The frozen paper evaluator binds to a slightly older model interface
(`camera_head_split_pose_branch`), which ships pinned under
`evaluation/vggt_compat/`; the training code at the repository root (`vggt/`)
is newer. This launcher pre-imports every dependency from the repository so
the evaluator's historical cluster path insert is inert and the correct model
code is selected automatically. Verified to reproduce the paper's Table 1-3
values exactly.

Usage: see scripts/eval_biwi.sh, or run with --self-test to check the
environment (requires torch/torchvision; see README Installation).
"""

from __future__ import annotations

import importlib
import inspect
import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "evaluation"
COMPARE = EVAL / "eval_biwi_crop_mtcnn_square_compare.py"

for entry in (ROOT, EVAL, EVAL / "vggt_compat"):
    entry = str(entry)
    if entry in sys.path:
        sys.path.remove(entry)
    sys.path.insert(0, entry)

PRE_IMPORTS = (
    "vggt",
    "vggt.models.vggt",
    "vggt.utils.pose_enc",
    "eval_biwi_crop",
    "eval_biwi_crop_mtcnn",
    "eval_biwi_crop_mtcnn_square",
    "eval_biwi_crop_mtcnn_square_translation",
    "eval_biwi_6dof_face",
    "prepare_biwi_for_sixdof_face",
    "revision_runs.detector_sensitivity.face_detectors",
)


def preimport() -> dict[str, str]:
    resolved = {}
    for name in PRE_IMPORTS:
        module = importlib.import_module(name)
        resolved[name] = getattr(module, "__file__", "<namespace>") or "<namespace>"
    return resolved


def self_test() -> int:
    resolved = preimport()
    from vggt.models.vggt import VGGT  # noqa: PLC0415

    assert "camera_head_split_pose_branch" in inspect.signature(VGGT.__init__).parameters
    print("self-test OK: evaluator model interface available")
    bad = 0
    for name, location in sorted(resolved.items()):
        inside = location == "<namespace>" or Path(location).resolve().is_relative_to(ROOT)
        print(f"  {name:55s} -> {location}{'' if inside else '  [OUTSIDE REPO]'}")
        bad += 0 if inside else 1
    print("self-test OK: all modules resolved inside the repository" if not bad
          else f"self-test FAILED: {bad} modules outside the repository")
    return 1 if bad else 0


def main() -> int:
    args = sys.argv[1:]
    if args and args[0] == "--self-test":
        return self_test()
    preimport()
    sys.argv = [str(COMPARE)] + args
    runpy.run_path(str(COMPARE), run_name="__main__")
    return 0


if __name__ == "__main__":
    sys.exit(main())
