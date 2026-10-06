"""Detector adapters used by the round-2 shared-crop sensitivity study."""

from revision_runs.detector_sensitivity.retinaface_adapter import (
    DEFAULT_RETINAFACE_MODEL,
    DEFAULT_RETINAFACE_REPO,
    RetinaFaceMTCNNAdapter,
)
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


class StrictFaceDetectionError(RuntimeError):
    """Raised when an independently processed frame has no valid face."""


def crop_face_strict(
    img_bgr,
    detector,
    _previous_box=None,
    crop_size=256,
    ad=0.4,
):
    """Crop one frame from its own detection, with no temporal or spatial fallback."""

    if img_bgr is None or img_bgr.size == 0:
        raise StrictFaceDetectionError("empty input image")

    image_height, image_width = img_bgr.shape[:2]
    image_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    boxes, probabilities = detector.detect(Image.fromarray(image_rgb))
    threshold = float(getattr(detector, "_strict_score_threshold", 0.95))

    candidates = []
    if boxes is not None and probabilities is not None:
        for box, probability in zip(boxes, probabilities):
            if probability is None or not np.isfinite(probability):
                continue
            box = np.asarray(box, dtype=np.float64)
            if box.shape != (4,) or not np.all(np.isfinite(box)):
                continue
            x1, y1, x2, y2 = box
            if float(probability) < threshold or x2 <= x1 or y2 <= y1:
                continue
            candidates.append((float(probability), box))

    if not candidates:
        raise StrictFaceDetectionError(
            f"no independent detection at confidence >= {threshold:.6f}"
        )

    _, (x1, y1, x2, y2) = max(candidates, key=lambda item: item[0])
    width = x2 - x1
    height = y2 - y1
    crop_box = [
        max(int(x1 - ad * width), 0),
        min(int(x2 + ad * width), image_width - 1),
        max(int(y1 - ad * height), 0),
        min(int(y2 + ad * height), image_height - 1),
    ]
    left, right, top, bottom = crop_box
    crop = img_bgr[top : bottom + 1, left : right + 1]
    if crop.size == 0:
        raise StrictFaceDetectionError(f"empty crop from detected box {crop_box}")
    return cv2.resize(crop, (crop_size, crop_size)), crop_box


class YuNetMTCNNAdapter:
    """Expose OpenCV YuNet through the ``facenet_pytorch.MTCNN.detect`` API.

    The legacy shared-crop helper applies an additional MTCNN probability gate.
    YuNet detections have already passed ``score_threshold``, so compatibility
    probabilities are one. Raw YuNet scores are recorded by the manifest pass.
    """

    def __init__(self, model_path, score_threshold=0.9, nms_threshold=0.3, top_k=5000):
        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"YuNet model not found: {model_path}")
        self.detector = cv2.FaceDetectorYN_create(
            str(model_path),
            "",
            (320, 320),
            float(score_threshold),
            float(nms_threshold),
            int(top_k),
        )

    def detect(self, pil_image):
        image_rgb = np.asarray(pil_image.convert("RGB"))
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        height, width = image_bgr.shape[:2]
        self.detector.setInputSize((width, height))
        _, faces = self.detector.detect(image_bgr)
        if faces is None or len(faces) == 0:
            return None, None

        boxes = np.empty((len(faces), 4), dtype=np.float32)
        boxes[:, 0] = faces[:, 0]
        boxes[:, 1] = faces[:, 1]
        boxes[:, 2] = faces[:, 0] + faces[:, 2]
        boxes[:, 3] = faces[:, 1] + faces[:, 3]
        return boxes, np.ones((len(faces),), dtype=np.float32)


def load_shared_face_detector(args, device):
    if args.face_detector == "mtcnn":
        from facenet_pytorch import MTCNN

        detector = MTCNN(keep_all=True, device=device)
        detector._strict_score_threshold = float(
            getattr(args, "mtcnn_score_threshold", 0.95)
        )
        return detector
    if args.face_detector == "yunet":
        detector = YuNetMTCNNAdapter(
            args.yunet_model,
            score_threshold=args.yunet_score_threshold,
            nms_threshold=args.yunet_nms_threshold,
        )
        detector._strict_score_threshold = 1.0
        return detector
    if args.face_detector == "retinaface":
        detector = RetinaFaceMTCNNAdapter(
            repo_path=args.retinaface_repo,
            model_path=args.retinaface_model,
            device=device,
            candidate_threshold=args.retinaface_candidate_threshold,
            nms_threshold=args.retinaface_nms_threshold,
        )
        detector._strict_score_threshold = float(args.retinaface_score_threshold)
        return detector
    raise ValueError(f"Unsupported face detector: {args.face_detector}")
