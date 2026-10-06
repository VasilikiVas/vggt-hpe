"""Isolated adapter for the upstream PyTorch RetinaFace implementation."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import cv2
import numpy as np


DEFAULT_RETINAFACE_REPO = Path(
    "/leonardo_scratch/fast/EUHPC_D32_089/external/Pytorch_Retinaface"
)
DEFAULT_RETINAFACE_MODEL = DEFAULT_RETINAFACE_REPO / "weights/Resnet50_Final.pth"


def _matches_upstream_package(name: str) -> bool:
    prefixes = ("data", "models", "layers", "utils")
    return any(name == prefix or name.startswith(f"{prefix}.") for prefix in prefixes)


def _import_upstream(repo_path: Path):
    """Import generic upstream packages without leaking them into the evaluator."""

    repo_path = str(repo_path.resolve())
    saved_modules = {
        name: module
        for name, module in list(sys.modules.items())
        if _matches_upstream_package(name)
    }
    for name in saved_modules:
        sys.modules.pop(name, None)

    imported_modules = set()
    sys.path.insert(0, repo_path)
    try:
        from data import cfg_re50
        from layers.functions.prior_box import PriorBox
        from models.retinaface import RetinaFace
        from utils.box_utils import decode
        from utils.nms.py_cpu_nms import py_cpu_nms

        imported_modules = {
            name for name in list(sys.modules) if _matches_upstream_package(name)
        }
    finally:
        if sys.path and sys.path[0] == repo_path:
            sys.path.pop(0)
        for name in imported_modules:
            sys.modules.pop(name, None)
        sys.modules.update(saved_modules)

    return cfg_re50, PriorBox, RetinaFace, decode, py_cpu_nms


class RetinaFaceMTCNNAdapter:
    """Expose upstream RetinaFace through facenet-pytorch's detect API."""

    def __init__(
        self,
        repo_path=DEFAULT_RETINAFACE_REPO,
        model_path=DEFAULT_RETINAFACE_MODEL,
        device="cuda",
        candidate_threshold=0.02,
        nms_threshold=0.4,
        top_k=5000,
        keep_top_k=750,
    ):
        import torch

        repo_path = Path(repo_path)
        model_path = Path(model_path)
        if not repo_path.is_dir():
            raise FileNotFoundError(f"RetinaFace repository not found: {repo_path}")
        if not model_path.is_file():
            raise FileNotFoundError(f"RetinaFace model not found: {model_path}")

        cfg, prior_box, model_class, decode, nms = _import_upstream(repo_path)
        self.cfg = copy.deepcopy(cfg)
        self.cfg["pretrain"] = False
        self.PriorBox = prior_box
        self.decode = decode
        self.nms = nms
        self.device = torch.device(device)
        self.candidate_threshold = float(candidate_threshold)
        self.nms_threshold = float(nms_threshold)
        self.top_k = int(top_k)
        self.keep_top_k = int(keep_top_k)

        self.model = model_class(cfg=self.cfg, phase="test")
        state = torch.load(model_path, map_location="cpu")
        state = state.get("state_dict", state)
        state = {
            key.removeprefix("module."): value for key, value in state.items()
        }
        used_keys = set(self.model.state_dict()).intersection(state)
        if not used_keys:
            raise RuntimeError(f"No RetinaFace checkpoint keys matched {model_path}")
        self.model.load_state_dict(state, strict=False)
        self.model.eval().to(self.device)

    def detect_bgr(self, image_bgr):
        import torch

        image = np.float32(image_bgr)
        height, width = image.shape[:2]
        scale = torch.tensor(
            [width, height, width, height],
            device=self.device,
            dtype=torch.float32,
        )
        image -= (104, 117, 123)
        tensor = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0)
        tensor = tensor.to(self.device)

        with torch.inference_mode():
            locations, confidences, _ = self.model(tensor)
            priors = self.PriorBox(
                self.cfg, image_size=(height, width)
            ).forward().to(self.device)
            boxes = self.decode(
                locations.squeeze(0), priors, self.cfg["variance"]
            )
            boxes = (boxes * scale).cpu().numpy()
            scores = confidences.squeeze(0)[:, 1].cpu().numpy()

        keep = np.flatnonzero(scores > self.candidate_threshold)
        boxes = boxes[keep]
        scores = scores[keep]
        order = scores.argsort()[::-1][: self.top_k]
        boxes = boxes[order]
        scores = scores[order]
        detections = np.hstack((boxes, scores[:, None])).astype(
            np.float32, copy=False
        )
        keep = self.nms(detections, self.nms_threshold)
        detections = detections[keep][: self.keep_top_k]
        if len(detections) == 0:
            return None, None
        return detections[:, :4], detections[:, 4]

    def detect(self, pil_image):
        image_rgb = np.asarray(pil_image.convert("RGB"))
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        return self.detect_bgr(image_bgr)
