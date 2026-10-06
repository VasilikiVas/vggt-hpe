import argparse
import csv
import gc
import glob
import importlib.util
import logging
import math
import os
import pickle
import re
import shutil
import sys
import types
from collections import defaultdict
from pathlib import Path

FILBY_DIR = "/leonardo_work/EUHPC_D32_089/head_pose/final_vggt_head_filby"
if FILBY_DIR not in sys.path:
    sys.path.insert(0, FILBY_DIR)

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

import eval_biwi_crop as base
import eval_biwi_crop_mtcnn_square as square_crop
import eval_biwi_crop_mtcnn_square_translation as vggt_translation
from eval_biwi_6dof_face import _build_patched_biwi_dataset_class, _build_safe_solvepnp_ransac
from prepare_biwi_for_sixdof_face import prepare_biwi_for_sixdof_face
from revision_runs.detector_sensitivity.face_detectors import load_shared_face_detector


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


TOKENHPE_TRANSFORM = base.TF.Compose(
    [
        base.TF.Resize(250),
        base.TF.CenterCrop(224),
        base.TF.ToTensor(),
        base.TF.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

TRG_TRANSFORM = base.TF.Compose(
    [
        base.TF.ToTensor(),
        base.TF.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def _apply_head_yz_flip_to_vggt_variants(variants):
    """Convert VGGT rotations to the FaMoS head_yz_flip scoring convention."""
    flipped = {}
    for name, R in variants.items():
        head_yz_flip = torch.diag(R.new_tensor([1.0, -1.0, -1.0]))
        flipped[name] = torch.matmul(R, head_yz_flip)
    return flipped


def _empty_sixdof_acc():
    return {"yaw": 0.0, "pitch": 0.0, "roll": 0.0, "tx": 0.0, "ty": 0.0, "tz": 0.0, "l2": 0.0, "count": 0}


def _method_display_name(method_key):
    names = {
        "vggt": "VGGT",
        "sixdrepnet": "6DRepNet",
        "tokenhpe": "TokenHPE",
        "whenet": "WHENet",
        "trg": "TRG",
    }
    return names.get(method_key, method_key)


def _parse_sixdof_subject(img_path):
    path = Path(img_path)
    if len(path.parts) < 3:
        return "unknown"
    return path.parts[-3]


def _build_random_anchor_candidates(frames, min_frame_gap):
    """
    Build candidate anchor indices for every target index.
    A candidate is valid if |frame_num(anchor)-frame_num(target)| >= min_frame_gap.
    """
    candidates = {}
    for target_idx, (target_frame_num, _, _) in enumerate(frames):
        valid = [
            anchor_idx
            for anchor_idx, (anchor_frame_num, _, _) in enumerate(frames)
            if abs(anchor_frame_num - target_frame_num) >= min_frame_gap
        ]
        if valid:
            candidates[target_idx] = valid
    return candidates


def _load_pair_subset_csv(pair_csv):
    pair_subset = defaultdict(list)

    def subject_aliases(value):
        raw = str(value).strip()
        aliases = [raw]
        if raw.isdigit():
            subject_int = int(raw)
            for width in (2, 3):
                padded = str(subject_int).zfill(width)
                if padded not in aliases:
                    aliases.append(padded)
        return aliases

    with open(pair_csv, newline="") as f:
        reader = csv.DictReader(f)
        required = {
            "subject",
            "anchor_frame_num",
            "anchor_pose_file",
            "query_frame_num",
            "query_pose_file",
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Pair CSV is missing required columns: {sorted(missing)}")

        for row in reader:
            item = {
                "anchor_frame_num": int(row["anchor_frame_num"]),
                "anchor_pose_file": row["anchor_pose_file"],
                "query_frame_num": int(row["query_frame_num"]),
                "query_pose_file": row["query_pose_file"],
            }
            for subj in subject_aliases(row["subject"]):
                pair_subset[subj].append(item)
    return pair_subset


def _resolve_subject_frame(frame_by_num, frame_by_pose, frame_num, pose_file, role, subj):
    frame = frame_by_num.get(frame_num)
    if frame is not None and os.path.basename(frame[2]) == pose_file:
        return frame

    frame = frame_by_pose.get(pose_file)
    if frame is not None and frame[0] == frame_num:
        return frame

    raise FileNotFoundError(
        f"Subject {subj}: could not resolve {role} frame_num={frame_num}, pose_file={pose_file}"
    )


def _build_subject_eval_groups_from_pair_csv(subj, frames, use_anchor_pair, pair_subset):
    if not pair_subset:
        return []

    frame_by_num = {frame_num: (frame_num, rgb_path, pose_path) for frame_num, rgb_path, pose_path in frames}
    frame_by_pose = {
        os.path.basename(pose_path): (frame_num, rgb_path, pose_path)
        for frame_num, rgb_path, pose_path in frames
    }

    if use_anchor_pair:
        grouped = defaultdict(list)
        for row in pair_subset:
            anchor_frame = _resolve_subject_frame(
                frame_by_num,
                frame_by_pose,
                row["anchor_frame_num"],
                row["anchor_pose_file"],
                role="anchor",
                subj=subj,
            )
            query_frame = _resolve_subject_frame(
                frame_by_num,
                frame_by_pose,
                row["query_frame_num"],
                row["query_pose_file"],
                role="query",
                subj=subj,
            )
            grouped[anchor_frame].append(query_frame)

        groups = []
        for anchor_frame, target_frames in sorted(grouped.items(), key=lambda item: item[0][0]):
            groups.append(
                {
                    "anchor": anchor_frame,
                    "targets": sorted(target_frames, key=lambda item: item[0]),
                }
            )
        return groups

    target_frames = [
        _resolve_subject_frame(
            frame_by_num,
            frame_by_pose,
            row["query_frame_num"],
            row["query_pose_file"],
            role="query",
            subj=subj,
        )
        for row in pair_subset
    ]
    return [{"anchor": None, "targets": sorted(target_frames, key=lambda item: item[0])}]


def _euler_from_rotation(R_t):
    """R_t: torch tensor shaped (3,3) in BIWI/pose.txt convention."""
    y, p, r = base.biwi_euler_deg_batch(R_t.unsqueeze(0))
    return y.item(), p.item(), r.item()


def _make_text_panel_like(img_rgb, lines):
    """Create an RGB panel with debug text, matching the target image size."""
    panel = np.full_like(img_rgb, 245)
    y = 34
    for line in lines:
        cv2.putText(
            panel,
            line,
            (14, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
        y += 30
    return panel


def _init_anchor_gap_hist(bin_edges_deg):
    edges = np.asarray(bin_edges_deg, dtype=np.float64)
    if edges.ndim != 1:
        raise ValueError("--anchor_gap_bins_deg must be a 1D list of numbers.")
    edges = np.unique(np.sort(edges))
    if edges.size < 2:
        raise ValueError("--anchor_gap_bins_deg must contain at least two distinct values.")
    if edges[0] > 0.0:
        edges = np.concatenate([[0.0], edges])
    return {
        "edges": edges,
        "count": np.zeros(edges.size - 1, dtype=np.int64),
        "vggt_mae_sum": np.zeros(edges.size - 1, dtype=np.float64),
        "vggt_l2_sum": np.zeros(edges.size - 1, dtype=np.float64),
        "sixd_mae_sum": np.zeros(edges.size - 1, dtype=np.float64),
        "sixd_count": np.zeros(edges.size - 1, dtype=np.int64),
        "tokenhpe_mae_sum": np.zeros(edges.size - 1, dtype=np.float64),
        "tokenhpe_count": np.zeros(edges.size - 1, dtype=np.int64),
        "whenet_mae_sum": np.zeros(edges.size - 1, dtype=np.float64),
        "whenet_count": np.zeros(edges.size - 1, dtype=np.int64),
        "trg_mae_sum": np.zeros(edges.size - 1, dtype=np.float64),
        "trg_count": np.zeros(edges.size - 1, dtype=np.int64),
    }


def _bin_index(value, edges):
    idx = int(np.searchsorted(edges, float(value), side="right") - 1)
    return max(0, min(idx, edges.size - 2))


def _rotation_gap_deg_batch(R_anchor_batch, R_target_batch):
    """
    Geodesic rotation gap (deg) between anchor and target in BIWI pose space.
    Inputs: tensors of shape (B,3,3).
    """
    R_delta = torch.matmul(R_target_batch, R_anchor_batch.transpose(-1, -2))
    trace = R_delta[:, 0, 0] + R_delta[:, 1, 1] + R_delta[:, 2, 2]
    cos_theta = ((trace - 1.0) * 0.5).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    return torch.acos(cos_theta) * (180.0 / np.pi)


def _update_anchor_gap_hist(
    hist_state,
    anchor_gap_deg,
    vggt_sample_mae,
    vggt_sample_l2,
    sixd_sample_mae=None,
    tokenhpe_sample_mae=None,
    whenet_sample_mae=None,
    trg_sample_mae=None,
):
    edges = hist_state["edges"]
    n = int(anchor_gap_deg.shape[0])
    for i in range(n):
        bi = _bin_index(anchor_gap_deg[i].item(), edges)
        hist_state["count"][bi] += 1
        hist_state["vggt_mae_sum"][bi] += float(vggt_sample_mae[i].item())
        hist_state["vggt_l2_sum"][bi] += float(vggt_sample_l2[i].item())
        if sixd_sample_mae is not None:
            v = float(sixd_sample_mae[i].item())
            if np.isfinite(v):
                hist_state["sixd_count"][bi] += 1
                hist_state["sixd_mae_sum"][bi] += v
        if tokenhpe_sample_mae is not None:
            v = float(tokenhpe_sample_mae[i].item())
            if np.isfinite(v):
                hist_state["tokenhpe_count"][bi] += 1
                hist_state["tokenhpe_mae_sum"][bi] += v
        if whenet_sample_mae is not None:
            v = float(whenet_sample_mae[i].item())
            if np.isfinite(v):
                hist_state["whenet_count"][bi] += 1
                hist_state["whenet_mae_sum"][bi] += v
        if trg_sample_mae is not None:
            v = float(trg_sample_mae[i].item())
            if np.isfinite(v):
                hist_state["trg_count"][bi] += 1
                hist_state["trg_mae_sum"][bi] += v


def _save_anchor_gap_histogram(hist_state, vis_dir, file_stem="anchor_gap_error_histogram", title_prefix=""):
    edges = hist_state["edges"]
    count = hist_state["count"]
    if int(count.sum()) == 0:
        return None, None

    vggt_mae = np.divide(
        hist_state["vggt_mae_sum"],
        count,
        out=np.full_like(hist_state["vggt_mae_sum"], np.nan),
        where=count > 0,
    )
    vggt_l2 = np.divide(
        hist_state["vggt_l2_sum"],
        count,
        out=np.full_like(hist_state["vggt_l2_sum"], np.nan),
        where=count > 0,
    )
    sixd_mae = np.divide(
        hist_state["sixd_mae_sum"],
        hist_state["sixd_count"],
        out=np.full_like(hist_state["sixd_mae_sum"], np.nan),
        where=hist_state["sixd_count"] > 0,
    )
    tokenhpe_mae = np.divide(
        hist_state["tokenhpe_mae_sum"],
        hist_state["tokenhpe_count"],
        out=np.full_like(hist_state["tokenhpe_mae_sum"], np.nan),
        where=hist_state["tokenhpe_count"] > 0,
    )
    whenet_mae = np.divide(
        hist_state["whenet_mae_sum"],
        hist_state["whenet_count"],
        out=np.full_like(hist_state["whenet_mae_sum"], np.nan),
        where=hist_state["whenet_count"] > 0,
    )
    trg_mae = np.divide(
        hist_state["trg_mae_sum"],
        hist_state["trg_count"],
        out=np.full_like(hist_state["trg_mae_sum"], np.nan),
        where=hist_state["trg_count"] > 0,
    )

    labels = [f"[{edges[i]:.1f}, {edges[i + 1]:.1f})" for i in range(edges.size - 1)]
    x = np.arange(len(labels))

    csv_path = os.path.join(vis_dir, f"{file_stem}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "bin",
                "gap_min_deg",
                "gap_max_deg",
                "count",
                "vggt_mean_mae_deg",
                "vggt_mean_l2_mm",
                "sixdrepnet_mean_mae_deg",
                "sixdrepnet_count",
                "tokenhpe_mean_mae_deg",
                "tokenhpe_count",
                "whenet_mean_mae_deg",
                "whenet_count",
                "trg_mean_mae_deg",
                "trg_count",
            ]
        )
        for i, label in enumerate(labels):
            w.writerow(
                [
                    label,
                    f"{edges[i]:.4f}",
                    f"{edges[i + 1]:.4f}",
                    int(count[i]),
                    f"{vggt_mae[i]:.6f}" if np.isfinite(vggt_mae[i]) else "",
                    f"{vggt_l2[i]:.6f}" if np.isfinite(vggt_l2[i]) else "",
                    f"{sixd_mae[i]:.6f}" if np.isfinite(sixd_mae[i]) else "",
                    int(hist_state["sixd_count"][i]),
                    f"{tokenhpe_mae[i]:.6f}" if np.isfinite(tokenhpe_mae[i]) else "",
                    int(hist_state["tokenhpe_count"][i]),
                    f"{whenet_mae[i]:.6f}" if np.isfinite(whenet_mae[i]) else "",
                    int(hist_state["whenet_count"][i]),
                    f"{trg_mae[i]:.6f}" if np.isfinite(trg_mae[i]) else "",
                    int(hist_state["trg_count"][i]),
                ]
            )

    fig_w = max(10, 0.75 * len(labels))
    fig, (ax_count, ax_err) = plt.subplots(2, 1, figsize=(fig_w, 8), sharex=True)
    ax_count.bar(x, count, color="#8fb8de")
    ax_count.set_ylabel("Samples")
    title = "Anchor-Target Rotation Gap Distribution"
    if title_prefix:
        title = f"{title_prefix} {title}"
    ax_count.set_title(title)
    ax_count.grid(axis="y", alpha=0.25)

    ax_err.plot(x, vggt_mae, marker="o", label="VGGT MAE (deg)")
    if np.isfinite(sixd_mae).any():
        ax_err.plot(x, sixd_mae, marker="o", label="6DRepNet MAE (deg)")
    if np.isfinite(tokenhpe_mae).any():
        ax_err.plot(x, tokenhpe_mae, marker="o", label="TokenHPE MAE (deg)")
    if np.isfinite(whenet_mae).any():
        ax_err.plot(x, whenet_mae, marker="o", label="WHENet MAE (deg)")
    if np.isfinite(trg_mae).any():
        ax_err.plot(x, trg_mae, marker="o", label="TRG MAE (deg)")
    ax_err.set_ylabel("Mean Error")
    ax_err.set_xlabel("Anchor-target GT rotation gap bin (deg)")
    ax_err.set_xticks(x)
    ax_err.set_xticklabels(labels, rotation=35, ha="right")
    ax_err.grid(axis="y", alpha=0.25)
    ax_err.legend(loc="best")

    png_path = os.path.join(vis_dir, f"{file_stem}.png")
    plt.tight_layout()
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return csv_path, png_path


def _format_hist_counts(hist_state):
    edges = hist_state["edges"]
    count = hist_state["count"]
    parts = []
    for i in range(edges.size - 1):
        parts.append(f"[{edges[i]:.0f},{edges[i + 1]:.0f}):{int(count[i])}")
    return "  ".join(parts)


def _scalar_or_nan(tensor_1d, idx):
    if tensor_1d is None:
        return float("nan")
    v = float(tensor_1d[idx].item())
    return v if np.isfinite(v) else float("nan")


def _save_pair_records(records, vis_dir, stem, metadata=None):
    """
    Save raw per-pair evaluation records so plots/bins can be rebuilt offline.
    Writes:
      - <stem>.csv
      - <stem>.pkl  (records + metadata)
    """
    if len(records) == 0:
        return None, None

    fields = [
        "subject",
        "target_frame",
        "anchor_frame",
        "anchor_frame_gap_abs",
        "anchor_selection",
        "anchor_gap_deg",
        "vggt_yaw_err_deg",
        "vggt_pitch_err_deg",
        "vggt_roll_err_deg",
        "vggt_mae_deg",
        "vggt_tx_err_mm",
        "vggt_ty_err_mm",
        "vggt_tz_err_mm",
        "vggt_t_l2_err_mm",
        "sixdrepnet_yaw_err_deg",
        "sixdrepnet_pitch_err_deg",
        "sixdrepnet_roll_err_deg",
        "sixdrepnet_mae_deg",
        "tokenhpe_yaw_err_deg",
        "tokenhpe_pitch_err_deg",
        "tokenhpe_roll_err_deg",
        "tokenhpe_mae_deg",
        "whenet_yaw_err_deg",
        "whenet_pitch_err_deg",
        "whenet_roll_err_deg",
        "whenet_mae_deg",
        "trg_yaw_err_deg",
        "trg_pitch_err_deg",
        "trg_roll_err_deg",
        "trg_mae_deg",
        "trg_tx_err_mm",
        "trg_ty_err_mm",
        "trg_tz_err_mm",
        "trg_t_l2_err_mm",
        "is_intersection",
    ]

    csv_path = os.path.join(vis_dir, f"{stem}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for rec in records:
            row = []
            for key in fields:
                val = rec.get(key, "")
                if isinstance(val, bool):
                    row.append(int(val))
                elif isinstance(val, (float, np.floating)):
                    row.append("" if not np.isfinite(float(val)) else f"{float(val):.8f}")
                else:
                    row.append(val)
            w.writerow(row)

    payload = {
        "fields": fields,
        "records": records,
        "metadata": metadata or {},
    }
    pkl_path = os.path.join(vis_dir, f"{stem}.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    return csv_path, pkl_path


def _resolve_sixdof_face_data_root(args):
    if args.sixdof_face_data_root:
        return args.sixdof_face_data_root
    if args.sixdof_face_cache_root:
        return args.sixdof_face_cache_root
    return os.path.join(args.sixdof_face_repo, "dataset", "ARKitFace")


def _load_module_from_path(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import module {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _update_after_crop_intrinsics(K, bbox_ltrb):
    """Shift principal point after cropping, K as [3,3], bbox=[l,t,r,b]."""
    K_ = K.copy()
    left, top, _, _ = bbox_ltrb
    K_[0, 2] = K_[0, 2] - float(left)
    K_[1, 2] = K_[1, 2] - float(top)
    return K_


def _update_after_resize_intrinsics(K, image_shape_hw, new_shape_hw):
    """Scale fx/fy/cx/cy after resize."""
    K_ = K.copy()
    in_h, in_w = float(image_shape_hw[0]), float(image_shape_hw[1])
    out_h, out_w = float(new_shape_hw[0]), float(new_shape_hw[1])
    sx = out_w / max(in_w, 1.0)
    sy = out_h / max(in_h, 1.0)
    K_[0, 0] *= sx
    K_[1, 1] *= sy
    K_[0, 2] *= sx
    K_[1, 2] *= sy
    return K_


def _build_trg_intrinsic_crop(K_rgb, bbox_info, img_size):
    """
    Build TRG-style K_crop tensor layout [4,3] (transposed), from BIWI K_rgb [3,3].
    bbox_info is [l,t,r,b,focal,h,w].
    """
    left, top, right, bottom = [float(v) for v in bbox_info[:4]]
    bbox_size = max(right - left, 1.0)
    K_tmp = _update_after_crop_intrinsics(K_rgb.astype(np.float32), [left, top, right, bottom])
    K_tmp = _update_after_resize_intrinsics(K_tmp, [bbox_size, bbox_size], [img_size, img_size])
    K_crop = np.zeros((4, 3), dtype=np.float32)
    K_crop[:3, :3] = K_tmp.T
    return K_crop


def _build_trg_square_bbox_from_mtcnn_box(box_xyxy, img_w, img_h):
    """
    Convert MTCNN box [x1,x2,y1,y2] to a square box and apply BIWI/TRG-style
    border handling by shifting (not shrinking) when the box exits image bounds.
    Returns integer (left, top, right, bottom).
    """
    x1, x2, y1, y2 = [float(v) for v in box_xyxy]
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    side = max(bw, bh)
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)

    left = int(round(cx - 0.5 * side))
    right = int(round(cx + 0.5 * side))
    top = int(round(cy - 0.5 * side))
    bottom = int(round(cy + 0.5 * side))

    # Match TRG BIWI dataset logic: preserve size by shifting window.
    if left < 0:
        off = -left
        left += off
        right += off
    if right > img_w - 1:
        off = right - (img_w - 1)
        left -= off
        right -= off
    if top < 0:
        off = -top
        top += off
        bottom += off
    if bottom > img_h - 1:
        off = bottom - (img_h - 1)
        top -= off
        bottom -= off

    left = int(max(0, left))
    top = int(max(0, top))
    right = int(min(img_w - 1, right))
    bottom = int(min(img_h - 1, bottom))

    if right <= left:
        right = min(img_w - 1, left + 1)
    if bottom <= top:
        bottom = min(img_h - 1, top + 1)
    return left, top, right, bottom


def _shift_bbox_inside_image_preserve_size(left, top, right, bottom, img_w, img_h):
    left = int(left)
    top = int(top)
    right = int(right)
    bottom = int(bottom)

    if left < 0:
        off = -left
        left += off
        right += off
    if right > img_w - 1:
        off = right - (img_w - 1)
        left -= off
        right -= off
    if top < 0:
        off = -top
        top += off
        bottom += off
    if bottom > img_h - 1:
        off = bottom - (img_h - 1)
        top -= off
        bottom -= off

    left = int(max(0, left))
    top = int(max(0, top))
    right = int(min(img_w - 1, right))
    bottom = int(min(img_h - 1, bottom))
    if right <= left:
        right = min(img_w - 1, left + 1)
    if bottom <= top:
        bottom = min(img_h - 1, top + 1)
    return left, top, right, bottom


def _build_trg_bbox_from_fan_kpt(pred_kpt_xy, img_w, img_h):
    """
    Replicate TRG BIWI dataset bbox logic from FAN landmarks:
    - center from min/max landmarks
    - size = max(w, h)
    - side_scale = 0.75 for each side
    - shift box to stay inside image while preserving window size
    """
    pts = np.asarray(pred_kpt_xy, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] < 2:
        raise ValueError("pred_kpt_xy must have shape [N,2] or [N,>=2]")
    xy = pts[:, :2]
    x_min, y_min = np.min(xy, axis=0)
    x_max, y_max = np.max(xy, axis=0)
    x_center = 0.5 * (x_min + x_max)
    y_center = 0.5 * (y_min + y_max)
    size = max(float(x_max - x_min), float(y_max - y_min), 1.0)
    side_scale = 0.75

    left = int(x_center - side_scale * size)
    right = int(x_center + side_scale * size)
    top = int(y_center - side_scale * size)
    bottom = int(y_center + side_scale * size)
    return _shift_bbox_inside_image_preserve_size(left, top, right, bottom, img_w, img_h)


def _trg_affine_crop(img_bgr, bbox_ltrb, img_size):
    """
    TRG-style crop: affine warp from full image box to fixed square resolution.
    """
    left, top, right, bottom = [float(v) for v in bbox_ltrb]
    src = np.float32(
        [
            [left, top],
            [left, bottom],
            [right, top],
        ]
    )
    dst = np.float32(
        [
            [0, 0],
            [0, img_size - 1],
            [img_size - 1, 0],
        ]
    )
    tform = cv2.getAffineTransform(src, dst)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return cv2.warpAffine(img_rgb, tform, (img_size, img_size))


def _resolve_trg_checkpoint_path(args):
    if args.trg_checkpoint:
        ckpt = os.path.abspath(args.trg_checkpoint)
        if not os.path.isfile(ckpt):
            raise FileNotFoundError(f"TRG checkpoint not found: {ckpt}")
        return ckpt

    repo_root = os.path.abspath(args.trg_repo)
    ckpt_root = os.path.abspath(args.trg_checkpoints_dir or os.path.join(repo_root, "checkpoint"))
    run_dir = os.path.join(ckpt_root, args.trg_name)
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"TRG run directory not found: {run_dir}")

    epoch_str = str(args.trg_epoch).strip()
    matches = []
    for name in sorted(os.listdir(run_dir)):
        if name.startswith("checkpoint-"):
            if epoch_str in {"", "latest"} or (f"checkpoint-{epoch_str}" in name):
                candidate = os.path.join(run_dir, name, "state_dict.bin")
                if os.path.isfile(candidate):
                    matches.append((name, candidate))

    if matches:
        if epoch_str == "latest":
            return matches[-1][1]
        return matches[0][1]

    direct = os.path.join(run_dir, "state_dict.bin")
    if os.path.isfile(direct):
        return direct

    raise FileNotFoundError(
        f"TRG checkpoint not found under {run_dir}. "
        f"Pass --trg_checkpoint, or set --trg_checkpoints_dir/--trg_name/--trg_epoch."
    )


def _patch_torchgeometry_bool_mask_compat():
    """
    TorchGeometry<=0.1 uses bool subtraction in rotation_matrix_to_quaternion:
        mask_c1 = mask_d2 * (1 - mask_d0_d1)
    which breaks on modern PyTorch.
    Apply the upstream project-recommended float-mask fix at runtime.
    """
    try:
        import torchgeometry as tgm
        from torchgeometry.core import conversions as tgm_conversions
    except Exception:
        return

    fn_name = getattr(tgm_conversions.rotation_matrix_to_quaternion, "__name__", "")
    if fn_name == "_rotation_matrix_to_quaternion_compat":
        return

    def _rotation_matrix_to_quaternion_compat(rotation_matrix, eps=1e-6):
        if not torch.is_tensor(rotation_matrix):
            raise TypeError(f"Input type is not a torch.Tensor. Got {type(rotation_matrix)}")
        if len(rotation_matrix.shape) > 3:
            raise ValueError(f"Input size must be a three dimensional tensor. Got {rotation_matrix.shape}")
        if rotation_matrix.shape[-2:] != (3, 4):
            raise ValueError(f"Input size must be a N x 3 x 4  tensor. Got {rotation_matrix.shape}")

        rmat_t = torch.transpose(rotation_matrix, 1, 2)

        mask_d2 = (rmat_t[:, 2, 2] < eps).float()
        mask_d0_d1 = (rmat_t[:, 0, 0] > rmat_t[:, 1, 1]).float()
        mask_d0_nd1 = (rmat_t[:, 0, 0] < -rmat_t[:, 1, 1]).float()

        t0 = 1 + rmat_t[:, 0, 0] - rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
        q0 = torch.stack(
            [
                rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
                t0,
                rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
                rmat_t[:, 2, 0] + rmat_t[:, 0, 2],
            ],
            -1,
        )
        t0_rep = t0.repeat(4, 1).t()

        t1 = 1 - rmat_t[:, 0, 0] + rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
        q1 = torch.stack(
            [
                rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
                rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
                t1,
                rmat_t[:, 1, 2] + rmat_t[:, 2, 1],
            ],
            -1,
        )
        t1_rep = t1.repeat(4, 1).t()

        t2 = 1 - rmat_t[:, 0, 0] - rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
        q2 = torch.stack(
            [
                rmat_t[:, 0, 1] - rmat_t[:, 1, 0],
                rmat_t[:, 2, 0] + rmat_t[:, 0, 2],
                rmat_t[:, 1, 2] + rmat_t[:, 2, 1],
                t2,
            ],
            -1,
        )
        t2_rep = t2.repeat(4, 1).t()

        t3 = 1 + rmat_t[:, 0, 0] + rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
        q3 = torch.stack(
            [
                t3,
                rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
                rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
                rmat_t[:, 0, 1] - rmat_t[:, 1, 0],
            ],
            -1,
        )
        t3_rep = t3.repeat(4, 1).t()

        mask_c0 = mask_d2 * mask_d0_d1
        mask_c1 = mask_d2 * (1 - mask_d0_d1)
        mask_c2 = (1 - mask_d2) * mask_d0_nd1
        mask_c3 = (1 - mask_d2) * (1 - mask_d0_nd1)
        mask_c0 = mask_c0.view(-1, 1).type_as(q0)
        mask_c1 = mask_c1.view(-1, 1).type_as(q1)
        mask_c2 = mask_c2.view(-1, 1).type_as(q2)
        mask_c3 = mask_c3.view(-1, 1).type_as(q3)

        q = q0 * mask_c0 + q1 * mask_c1 + q2 * mask_c2 + q3 * mask_c3
        q /= torch.sqrt(t0_rep * mask_c0 + t1_rep * mask_c1 + t2_rep * mask_c2 + t3_rep * mask_c3)
        q *= 0.5
        return q

    tgm_conversions.rotation_matrix_to_quaternion = _rotation_matrix_to_quaternion_compat
    # Keep top-level aliases consistent as well.
    tgm.rotation_matrix_to_quaternion = _rotation_matrix_to_quaternion_compat
    logging.info("Applied torchgeometry bool-mask compatibility patch for TRG.")


def load_trg_model(args, device):
    if args.no_trg:
        return None

    repo_root = os.path.abspath(args.trg_repo)
    if not os.path.isdir(repo_root):
        raise FileNotFoundError(f"TRG repo not found: {repo_root}")

    cfg_path = os.path.join(repo_root, "models", "trg_model", "configs", "trg_face_config.yaml")
    init_path = os.path.join(repo_root, "data", "init_vtx_cam.pkl")
    subsample_path = os.path.join(repo_root, "data", "arkit_subsample.pkl")
    for p in [cfg_path, init_path, subsample_path]:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"TRG file missing: {p}")

    ckpt_path = _resolve_trg_checkpoint_path(args)

    old_cwd = os.getcwd()
    old_sys_path = list(sys.path)

    # Keep conflicting module namespaces isolated from prior baselines (e.g., 6DoF Face "models").
    old_modules = {}
    prefixes = ("models", "data", "options", "util")
    for name in list(sys.modules.keys()):
        if any(name == p or name.startswith(f"{p}.") for p in prefixes):
            old_modules[name] = sys.modules.pop(name)

    import torch.utils.model_zoo as model_zoo

    old_load_url = model_zoo.load_url
    warned = {"done": False}

    def _offline_load_url(url, *args_, **kwargs_):
        model_dir = kwargs_.get("model_dir", None)
        if model_dir is not None:
            filename = os.path.basename(url)
            local_path = os.path.join(model_dir, filename)
            if os.path.isfile(local_path):
                return torch.load(local_path, map_location="cpu")
        if not warned["done"]:
            logging.warning("TRG: skipping online ResNet preload (offline environment).")
            warned["done"] = True
        return {}

    try:
        model_zoo.load_url = _offline_load_url
        _patch_torchgeometry_bool_mask_compat()
        os.chdir(repo_root)
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)

        from models.trg_model.core.cfgs import parse_args as parse_args_trg
        from models.trg_model import load_trg
        from util.pkl import read_pkl

        cfg = parse_args_trg(cfg_path)
        init_face_pose = read_pkl("data/init_vtx_cam.pkl")
        init_face = init_face_pose["t_vtx_1220"]
        init_pose = init_face_pose["R_t"]

        model = load_trg(cfg, init_face, init_pose).to(device)

        logging.info(f"Loading TRG: {ckpt_path}")
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if any(k.startswith("module.") for k in state.keys()):
            state = {k.replace("module.", "", 1): v for k, v in state.items()}
        miss, unex = model.load_state_dict(state, strict=False)
        logging.info(f"  TRG — missing: {len(miss)}, unexpected: {len(unex)}")
        model.eval()
        return model

    finally:
        model_zoo.load_url = old_load_url
        os.chdir(old_cwd)
        sys.path = old_sys_path
        # Restore prior namespaces so downstream imports behave exactly as before.
        for name in list(sys.modules.keys()):
            if any(name == p or name.startswith(f"{p}.") for p in prefixes):
                sys.modules.pop(name, None)
        sys.modules.update(old_modules)


def load_tokenhpe_model(args, device):
    if args.no_tokenhpe:
        return None

    repo_root = os.path.abspath(args.tokenhpe_repo)
    ckpt_path = os.path.abspath(args.tokenhpe_checkpoint)
    model_py = os.path.join(repo_root, "model.py")
    vit_py = os.path.join(repo_root, "ViT_model.py")
    utils_py = os.path.join(repo_root, "utils.py")

    if not os.path.isdir(repo_root):
        raise FileNotFoundError(f"TokenHPE repo not found: {repo_root}")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"TokenHPE checkpoint not found: {ckpt_path}")
    for p in [model_py, vit_py, utils_py]:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"TokenHPE file missing: {p}")

    # TokenHPE imports timm's trunc_normal_ helper. If timm is unavailable in the
    # current env, provide a tiny shim backed by torch.nn.init.trunc_normal_.
    try:
        import timm  # noqa: F401
    except Exception:
        import torch.nn.init as nn_init

        timm_mod = types.ModuleType("timm")
        timm_models_mod = types.ModuleType("timm.models")
        timm_layers_mod = types.ModuleType("timm.models.layers")
        timm_weight_init_mod = types.ModuleType("timm.models.layers.weight_init")
        timm_weight_init_mod.trunc_normal_ = nn_init.trunc_normal_
        timm_layers_mod.weight_init = timm_weight_init_mod
        timm_models_mod.layers = timm_layers_mod
        timm_mod.models = timm_models_mod
        sys.modules.setdefault("timm", timm_mod)
        sys.modules.setdefault("timm.models", timm_models_mod)
        sys.modules.setdefault("timm.models.layers", timm_layers_mod)
        sys.modules.setdefault("timm.models.layers.weight_init", timm_weight_init_mod)

    try:
        import seaborn  # noqa: F401
    except Exception:
        seaborn_mod = types.ModuleType("seaborn")
        seaborn_mod.set = lambda *args, **kwargs: None
        seaborn_mod.heatmap = lambda *args, **kwargs: None
        sys.modules.setdefault("seaborn", seaborn_mod)

    old_model = sys.modules.get("model")
    old_vit = sys.modules.get("ViT_model")
    old_utils = sys.modules.get("utils")
    try:
        sys.modules["ViT_model"] = _load_module_from_path("ViT_model", vit_py)
        sys.modules["utils"] = _load_module_from_path("utils", utils_py)
        token_model_module = _load_module_from_path("_tokenhpe_model_local", model_py)
    finally:
        if old_model is not None:
            sys.modules["model"] = old_model
        else:
            sys.modules.pop("model", None)
        if old_vit is not None:
            sys.modules["ViT_model"] = old_vit
        else:
            sys.modules.pop("ViT_model", None)
        if old_utils is not None:
            sys.modules["utils"] = old_utils
        else:
            sys.modules.pop("utils", None)

    # TokenHPE's original helper defaults to use_gpu=True.
    # Override to avoid CPU-only crashes on login nodes.
    if hasattr(token_model_module, "compute_rotation_matrix_from_ortho6d"):
        _orig_compute = token_model_module.compute_rotation_matrix_from_ortho6d

        def _compute_rotation_matrix_from_ortho6d_safe(poses, use_gpu=None):
            if use_gpu is None:
                use_gpu = torch.cuda.is_available()
            return _orig_compute(poses, use_gpu=use_gpu)

        token_model_module.compute_rotation_matrix_from_ortho6d = _compute_rotation_matrix_from_ortho6d_safe

    TokenHPE = token_model_module.TokenHPE
    model = TokenHPE(
        num_ori_tokens=9,
        depth=3,
        heads=8,
        embedding="sine",
        dim=128,
    )

    logging.info(f"Loading TokenHPE: {ckpt_path}")
    saved = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = saved.get("model_state_dict", saved)
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    miss, unex = model.load_state_dict(state, strict=False)
    logging.info(f"  TokenHPE — missing: {len(miss)}, unexpected: {len(unex)}")
    model.eval().to(device)
    return model


def load_whenet_model(args):
    if args.no_whenet:
        return None

    repo_root = os.path.abspath(args.whenet_repo)
    ckpt_path = os.path.abspath(args.whenet_checkpoint)
    whenet_py = os.path.join(repo_root, "whenet.py")
    utils_py = os.path.join(repo_root, "utils.py")

    if not os.path.isdir(repo_root):
        raise FileNotFoundError(f"WHENet repo not found: {repo_root}")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"WHENet checkpoint not found: {ckpt_path}")
    for p in [whenet_py, utils_py]:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"WHENet file missing: {p}")

    # WHENet imports `efficientnet` and standalone `keras`.
    # In many envs we only have TensorFlow (tf.keras) and not the legacy efficientnet package.
    # Provide a lightweight compatibility shim so WHENet can initialize without extra installs.
    old_utils = sys.modules.get("utils")
    old_keras = sys.modules.get("keras")
    old_efficientnet = sys.modules.get("efficientnet")
    try:
        try:
            import keras  # noqa: F401
        except Exception:
            import tensorflow as tf

            keras_mod = types.ModuleType("keras")
            keras_mod.layers = tf.keras.layers
            keras_mod.models = tf.keras.models
            keras_mod.backend = tf.keras.backend
            keras_mod.initializers = tf.keras.initializers
            keras_mod.regularizers = tf.keras.regularizers
            keras_mod.constraints = tf.keras.constraints
            keras_mod.optimizers = tf.keras.optimizers
            sys.modules["keras"] = keras_mod

        try:
            import efficientnet  # noqa: F401
        except Exception:
            import tensorflow as tf

            efficientnet_mod = types.ModuleType("efficientnet")

            def _efficientnet_b0_compat(*args, **kwargs):
                kwargs.setdefault("weights", None)
                return tf.keras.applications.EfficientNetB0(*args, **kwargs)

            efficientnet_mod.EfficientNetB0 = _efficientnet_b0_compat
            sys.modules["efficientnet"] = efficientnet_mod

        sys.modules["utils"] = _load_module_from_path("utils", utils_py)
        whenet_module = _load_module_from_path("_whenet_local", whenet_py)
    finally:
        if old_utils is not None:
            sys.modules["utils"] = old_utils
        else:
            sys.modules.pop("utils", None)
        if old_keras is not None:
            sys.modules["keras"] = old_keras
        else:
            sys.modules.pop("keras", None)
        if old_efficientnet is not None:
            sys.modules["efficientnet"] = old_efficientnet
        else:
            sys.modules.pop("efficientnet", None)

    WHENet = whenet_module.WHENet
    logging.info(f"Loading WHENet: {ckpt_path}")
    model = WHENet(ckpt_path)
    if hasattr(model, "model") and hasattr(model.model, "predict"):
        _orig_predict = model.model.predict

        def _predict_quiet(*args, **kwargs):
            kwargs.setdefault("verbose", 0)
            return _orig_predict(*args, **kwargs)

        model.model.predict = _predict_quiet
    return model


def _angular_error_wrap_deg(gt_deg, pred_deg):
    return torch.min(
        torch.stack(
            (
                torch.abs(gt_deg - pred_deg),
                torch.abs(pred_deg + 360 - gt_deg),
                torch.abs(pred_deg - 360 - gt_deg),
                torch.abs(pred_deg + 180 - gt_deg),
                torch.abs(pred_deg - 180 - gt_deg),
            )
        ),
        0,
    )[0]


def _token_pose_deg_from_biwi_pose_txt(pose_path):
    """
    Reproduce 6DRepNet/TokenHPE BIWI preprocessing pose conversion in
    TYY_create_db_biwi_70_30.py (stored as [yaw, pitch, roll] in degrees).
    """
    mat = np.loadtxt(pose_path, dtype=np.float64)
    if mat.ndim == 1:
        mat = mat.reshape(4, 3)
    R = mat[:3, :].T
    roll = -math.atan2(R[1, 0], R[0, 0]) * (180.0 / np.pi)
    yaw = -math.atan2(-R[2, 0], math.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2)) * (180.0 / np.pi)
    pitch = math.atan2(R[2, 1], R[2, 2]) * (180.0 / np.pi)
    return np.array([yaw, pitch, roll], dtype=np.float64)


def _recover_tokenhpe_native_sample_keys(args, npz_path):
    """
    Recover (subject, frame_num) for each BIWI_test.npz sample by matching
    native pose sequence against BIWI pose.txt sequences per subject.
    This works because the original script concatenates full selected subjects
    in ascending subject id order.
    """
    npz = np.load(npz_path, mmap_mode="r")
    if "pose" not in npz:
        raise RuntimeError(f"TokenHPE native npz missing 'pose' array: {npz_path}")
    pose_seq = np.asarray(npz["pose"], dtype=np.float64)
    n_total = int(pose_seq.shape[0])
    cache_path = os.path.join(os.path.dirname(npz_path), f".{Path(npz_path).stem}_sample_keys_cache.csv")

    if os.path.isfile(cache_path):
        try:
            cached = []
            with open(cache_path, "r", newline="") as f:
                r = csv.reader(f)
                header = next(r, None)
                if header != ["subject", "frame_num"]:
                    raise ValueError("cache header mismatch")
                for row in r:
                    if len(row) != 2:
                        continue
                    cached.append((row[0], int(row[1])))
            if len(cached) == n_total:
                logging.info("[TokenHPE native] loaded sample mapping cache: %s", cache_path)
                return cached
        except Exception as e:
            logging.warning("[TokenHPE native] mapping cache read failed (%s), rebuilding.", e)

    subject_records = []
    for s in range(1, 25):
        subj = f"{s:02d}"
        subj_dir = os.path.join(args.biwi_dir, subj)
        if not os.path.isdir(subj_dir):
            continue
        frames = base.get_subject_frames(subj_dir)
        if len(frames) == 0:
            continue
        subj_pose = np.stack(
            [_token_pose_deg_from_biwi_pose_txt(pose_path) for _, _, pose_path in frames],
            axis=0,
        )
        subject_records.append(
            {
                "subj": subj,
                "frames": [int(frame_num) for frame_num, _, _ in frames],
                "pose": subj_pose,
                "n": int(subj_pose.shape[0]),
            }
        )

    cursor = 0
    sample_keys = []
    selected_subjects = []
    remaining = {rec["subj"]: rec for rec in subject_records}

    while cursor < n_total:
        matched_subj = None
        for subj in list(remaining.keys()):
            rec = remaining[subj]
            n_subj = rec["n"]
            if cursor + n_subj > n_total:
                continue
            if np.allclose(pose_seq[cursor : cursor + n_subj], rec["pose"], atol=1e-6, rtol=0.0):
                matched_subj = subj
                break

        if matched_subj is None:
            break

        rec = remaining.pop(matched_subj)
        selected_subjects.append(matched_subj)
        sample_keys.extend([(matched_subj, frame_num) for frame_num in rec["frames"]])
        cursor += rec["n"]

    if cursor != n_total or len(sample_keys) != n_total:
        raise RuntimeError(
            "Could not recover TokenHPE native BIWI sample->frame mapping from pose sequence.\n"
            f"Recovered {cursor}/{n_total} samples. "
            "Ensure --biwi_dir matches the BIWI root used to build --tokenhpe_native_npz."
        )

    logging.info(
        "[TokenHPE native] recovered split: %d samples across %d subjects: %s",
        n_total,
        len(selected_subjects),
        ", ".join(selected_subjects),
    )
    try:
        with open(cache_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["subject", "frame_num"])
            for subj, frame_num in sample_keys:
                w.writerow([subj, frame_num])
        logging.info("[TokenHPE native] wrote sample mapping cache: %s", cache_path)
    except Exception as e:
        logging.warning("[TokenHPE native] could not write mapping cache (%s).", e)
    return sample_keys


def _parse_biwi_img_key(path_like):
    key = str(path_like).replace("\\", "/")
    m = re.search(r"([0-9]{2})/frame_([0-9]+)_rgb\.png$", key)
    if m is None:
        return None, None
    return m.group(1), int(m.group(2))


def evaluate_tokenhpe_native(args, device):
    """
    Native TokenHPE BIWI evaluation protocol:
    - load BIWI npz
    - transforms: Resize(250) -> CenterCrop(224) -> ImageNet norm
    - metric computation with BIWI label convention [pitch, yaw, roll]
    """
    if args.no_tokenhpe:
        return None

    npz_path = os.path.abspath(args.tokenhpe_native_npz)
    if not os.path.isfile(npz_path):
        raise FileNotFoundError(
            f"TokenHPE native BIWI npz not found: {npz_path}\n"
            f"Pass --tokenhpe_native_npz (e.g. BIWI_test.npz prepared with official 6DRepNet/TokenHPE preprocessing)."
        )

    repo_root = os.path.abspath(args.tokenhpe_repo)
    datasets_py = os.path.join(repo_root, "datasets.py")
    utils_py = os.path.join(repo_root, "utils.py")
    for p in [datasets_py, utils_py]:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"TokenHPE file missing: {p}")

    tokenhpe_model = load_tokenhpe_model(args, device)
    sample_keys = _recover_tokenhpe_native_sample_keys(args, npz_path)

    old_utils = sys.modules.get("utils")
    try:
        sys.modules["utils"] = _load_module_from_path("utils", utils_py)
        token_datasets = _load_module_from_path("_tokenhpe_datasets_local", datasets_py)
        token_utils = sys.modules["utils"]
    finally:
        if old_utils is not None:
            sys.modules["utils"] = old_utils
        else:
            sys.modules.pop("utils", None)

    transformations = base.TF.Compose(
        [
            base.TF.Resize(250),
            base.TF.CenterCrop(224),
            base.TF.ToTensor(),
            base.TF.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    dataset = token_datasets.getDataset(
        "BIWI",
        "",
        npz_path,
        transformations,
        train_mode=False,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=args.tokenhpe_native_batch_size,
        num_workers=args.tokenhpe_native_num_workers,
        shuffle=False,
        drop_last=False,
    )

    yaw_error = pitch_error = roll_error = 0.0
    total = 0
    start_idx = 0
    per_frame = {}
    use_gpu_euler = bool(device.type == "cuda" and torch.cuda.is_available())

    with torch.no_grad():
        for batch_idx, (images, _, cont_labels, _) in enumerate(dataloader):
            if batch_idx % 20 == 0:
                logging.info(f"[TokenHPE native] batch {batch_idx + 1}/{len(dataloader)}")

            images = images.to(device)
            R_pred, _ = tokenhpe_model(images)
            R_pred_cpu = R_pred.detach().cpu()
            euler = token_utils.compute_euler_angles_from_rotation_matrices(
                R_pred,
                use_gpu=use_gpu_euler,
            ) * (180.0 / np.pi)
            euler = euler.detach().cpu()

            # BIWI-space conversion used in shared compare loop and visualization.
            y_biwi, p_biwi, r_biwi, R_biwi = base.decode_sixdrepnet_pose(R_pred_cpu)

            p_pred_deg = euler[:, 0]
            y_pred_deg = euler[:, 1]
            r_pred_deg = euler[:, 2]

            # BIWI dataset stores cont_labels as [pitch, yaw, roll] (radians).
            p_gt_deg = cont_labels[:, 0].float() * (180.0 / np.pi)
            y_gt_deg = cont_labels[:, 1].float() * (180.0 / np.pi)
            r_gt_deg = cont_labels[:, 2].float() * (180.0 / np.pi)

            yaw_error += _angular_error_wrap_deg(y_gt_deg, y_pred_deg).sum().item()
            pitch_error += _angular_error_wrap_deg(p_gt_deg, p_pred_deg).sum().item()
            roll_error += _angular_error_wrap_deg(r_gt_deg, r_pred_deg).sum().item()
            bs = int(images.shape[0])
            total += bs

            for li in range(bs):
                gi = start_idx + li
                if gi >= len(sample_keys):
                    break
                subj, frame_num = sample_keys[gi]
                per_frame[(subj, frame_num)] = {
                    "euler": (
                        float(y_biwi[li].item()),
                        float(p_biwi[li].item()),
                        float(r_biwi[li].item()),
                    ),
                    "R_biwi": np.asarray(R_biwi[li], dtype=np.float32),
                }
            start_idx += bs

    if total == 0:
        raise RuntimeError("TokenHPE native evaluation loaded 0 samples.")

    overall = {
        "yaw": yaw_error / total,
        "pitch": pitch_error / total,
        "roll": roll_error / total,
        "mean": (yaw_error + pitch_error + roll_error) / (3.0 * total),
        "count": total,
    }
    logging.info(
        "[TokenHPE native] overall — "
        f"Yaw: {overall['yaw']:.4f}  Pitch: {overall['pitch']:.4f}  "
        f"Roll: {overall['roll']:.4f}  MAE: {overall['mean']:.4f}  Count: {overall['count']}"
    )
    return {
        "overall": overall,
        "per_frame": per_frame,
    }


def _resolve_trg_native_annot_path(args, repo_root):
    candidates = []
    if args.trg_native_annot:
        candidates.append(os.path.abspath(args.trg_native_annot))
    candidates.extend(
        [
            os.path.join(repo_root, "data", "annot_mtcnn_fan.pkl"),
            os.path.join(
                repo_root,
                "dataset",
                "BIWI",
                "download_from_official_site",
                "kinect_head_pose_db",
                "hpdb",
                "annot_mtcnn_fan.pkl",
            ),
        ]
    )
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        "TRG native annotation file not found. Checked:\n  - "
        + "\n  - ".join(candidates)
        + "\nPlease provide --trg_native_annot or place annot_mtcnn_fan.pkl in TRG-Release/data/."
    )


def _load_trg_fan_keypoints_map(args):
    repo_root = os.path.abspath(args.trg_repo)
    annot_path = _resolve_trg_native_annot_path(args, repo_root)
    with open(annot_path, "rb") as f:
        annot = pickle.load(f)
    pred_kpt_map_raw = annot.get("pred_kpt", None)
    if not isinstance(pred_kpt_map_raw, dict):
        raise ValueError(
            f"TRG annotation file has no valid 'pred_kpt' dict: {annot_path}"
        )
    pred_kpt_map = {}
    for k, v in pred_kpt_map_raw.items():
        pred_kpt_map[str(k).replace("\\", "/")] = np.asarray(v, dtype=np.float32)
    return pred_kpt_map, annot_path


def _safe_symlink_or_copy(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.lexists(dst):
        return
    try:
        os.symlink(src, dst)
    except OSError:
        if os.path.isdir(src):
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)


def _prepare_trg_native_dataset_layout(args, repo_root, annot_src_path):
    """
    Prepare TRG's hardcoded BIWI path:
      ./dataset/BIWI/download_from_official_site/kinect_head_pose_db/hpdb
    without modifying the original BIWI root. Returns cleanup state.
    """
    biwi_root = os.path.abspath(args.biwi_dir)
    if not os.path.isdir(biwi_root):
        raise FileNotFoundError(f"--biwi_dir does not exist: {biwi_root}")

    dataset_dir = os.path.join(repo_root, "dataset")
    os.makedirs(dataset_dir, exist_ok=True)
    dataset_biwi_link = os.path.join(dataset_dir, "BIWI")

    native_root = os.path.join(dataset_dir, "__biwi_native_layout_codex")
    hpdb_dir = os.path.join(native_root, "download_from_official_site", "kinect_head_pose_db", "hpdb")
    os.makedirs(hpdb_dir, exist_ok=True)

    # Mirror BIWI tree via top-level symlinks (subjects, objs, etc.).
    for name in sorted(os.listdir(biwi_root)):
        src = os.path.join(biwi_root, name)
        dst = os.path.join(hpdb_dir, name)
        if not os.path.lexists(dst):
            _safe_symlink_or_copy(src, dst)

    annot_dst = os.path.join(hpdb_dir, "annot_mtcnn_fan.pkl")
    if os.path.lexists(annot_dst):
        try:
            same = os.path.realpath(annot_dst) == os.path.realpath(annot_src_path)
        except OSError:
            same = False
        if not same:
            os.remove(annot_dst)
            _safe_symlink_or_copy(annot_src_path, annot_dst)
    else:
        _safe_symlink_or_copy(annot_src_path, annot_dst)

    backup_path = None
    if os.path.lexists(dataset_biwi_link):
        backup_path = f"{dataset_biwi_link}.__orig_codex_{os.getpid()}"
        if os.path.lexists(backup_path):
            if os.path.isdir(backup_path) and not os.path.islink(backup_path):
                shutil.rmtree(backup_path)
            else:
                os.remove(backup_path)
        os.rename(dataset_biwi_link, backup_path)
    os.symlink(native_root, dataset_biwi_link)

    logging.info("TRG native BIWI layout ready at %s", hpdb_dir)
    logging.info("TRG native annot source: %s", annot_src_path)
    return {
        "dataset_biwi_link": dataset_biwi_link,
        "backup_path": backup_path,
        "native_root": native_root,
    }


def _restore_trg_native_dataset_layout(state):
    dataset_biwi_link = state["dataset_biwi_link"]
    backup_path = state["backup_path"]
    try:
        if os.path.islink(dataset_biwi_link):
            os.remove(dataset_biwi_link)
        elif os.path.isdir(dataset_biwi_link):
            shutil.rmtree(dataset_biwi_link)
        elif os.path.exists(dataset_biwi_link):
            os.remove(dataset_biwi_link)
    finally:
        if backup_path and os.path.lexists(backup_path):
            os.rename(backup_path, dataset_biwi_link)


def _run_trg_native_detailed_inprocess(args, repo_root, ckpt_path):
    old_cwd = os.getcwd()
    old_sys_path = list(sys.path)

    old_modules = {}
    prefixes = ("models", "data", "options", "util")
    for name in list(sys.modules.keys()):
        if any(name == p or name.startswith(f"{p}.") for p in prefixes):
            old_modules[name] = sys.modules.pop(name)

    import torch.utils.model_zoo as model_zoo

    old_load_url = model_zoo.load_url
    warned = {"done": False}

    def _offline_load_url(url, *args_, **kwargs_):
        model_dir = kwargs_.get("model_dir", None)
        if model_dir is not None:
            filename = os.path.basename(url)
            local_path = os.path.join(model_dir, filename)
            if os.path.isfile(local_path):
                return torch.load(local_path, map_location="cpu")
        if not warned["done"]:
            logging.warning("TRG native: skipping online ResNet preload (offline environment).")
            warned["done"] = True
        return {}

    try:
        model_zoo.load_url = _offline_load_url
        _patch_torchgeometry_bool_mask_compat()
        os.chdir(repo_root)
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)

        from data.biwi_dataset import BIWIDataset
        from models.trg_model import load_trg
        from models.trg_model.core.cfgs import parse_args as parse_args_trg
        from util.metric import calc_rotation_mae, calc_trans_mae
        from util.pkl import read_pkl

        device = torch.device(f"cuda:{args.gpu}")
        cfg_path = os.path.join(repo_root, "models", "trg_model", "configs", "trg_face_config.yaml")
        cfg = parse_args_trg(cfg_path)
        init_face_pose = read_pkl("data/init_vtx_cam.pkl")
        init_face = init_face_pose["t_vtx_1220"]
        init_pose = init_face_pose["R_t"]
        model = load_trg(cfg, init_face, init_pose).to(device)

        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if any(k.startswith("module.") for k in state.keys()):
            state = {k.replace("module.", "", 1): v for k, v in state.items()}
        miss, unex = model.load_state_dict(state, strict=False)
        logging.info(f"[TRG native] load checkpoint: {ckpt_path}")
        logging.info(f"[TRG native] missing: {len(miss)}, unexpected: {len(unex)}")
        model.eval()

        opt = types.SimpleNamespace(isTrain=False, img_size=args.trg_img_size)
        dataset = BIWIDataset(opt)
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=args.trg_native_batch_size,
            num_workers=args.trg_native_num_workers,
            shuffle=False,
            drop_last=True,
        )

        cal_cache = {}

        def get_cal(subj):
            if subj not in cal_cache:
                cal_cache[subj] = base.load_rgb_intrinsics(os.path.join(args.biwi_dir, subj, "rgb.cal"))
            return cal_cache[subj]

        sums = {"yaw": 0.0, "pitch": 0.0, "roll": 0.0, "tx": 0.0, "ty": 0.0, "tz": 0.0, "count": 0}
        per_subject_acc = defaultdict(_empty_sixdof_acc)
        per_frame = {}

        with torch.no_grad():
            for bi, data in enumerate(dataloader):
                if bi % 20 == 0:
                    logging.info(f"[TRG native] batch {bi + 1}/{len(dataloader)}")

                img = data["img"].to(device)
                gt_extrinsic = data["R_t"].to(device)
                intrinsic = data["K_crop"].to(device)
                bbox_info = data["bbox_info"].to(device)

                preds_dict, _ = model(img, intrinsic, bbox_info)
                pred_R_t = preds_dict["output"][-1]["pred_R_t"].detach().cpu()
                pred_M = pred_R_t.transpose(1, 2).numpy()
                gt_M = gt_extrinsic.transpose(1, 2).detach().cpu().numpy()
                img_keys = data["img_path_key"]

                for li in range(pred_M.shape[0]):
                    subj, frame_num = _parse_biwi_img_key(img_keys[li])
                    if subj is None:
                        continue

                    R_pred_rgb = pred_M[li, :3, :3]
                    t_pred_rgb_m = pred_M[li, :3, 3]
                    R_gt_rgb = gt_M[li, :3, :3]
                    t_gt_rgb_m = gt_M[li, :3, 3]

                    y_arr, p_arr, r_arr = calc_rotation_mae(R_pred_rgb[None], R_gt_rgb[None])
                    y_e = float(y_arr[0])
                    p_e = float(p_arr[0])
                    r_e = float(r_arr[0])
                    tx_e_m, ty_e_m, tz_e_m = calc_trans_mae(t_pred_rgb_m[None], t_gt_rgb_m[None])
                    tx_e = float(tx_e_m[0] * 1000.0)
                    ty_e = float(ty_e_m[0] * 1000.0)
                    tz_e = float(tz_e_m[0] * 1000.0)
                    l2_e = float(np.sqrt(tx_e * tx_e + ty_e * ty_e + tz_e * tz_e))

                    sums["yaw"] += y_e
                    sums["pitch"] += p_e
                    sums["roll"] += r_e
                    sums["tx"] += tx_e
                    sums["ty"] += ty_e
                    sums["tz"] += tz_e
                    sums["count"] += 1

                    acc = per_subject_acc[subj]
                    acc["yaw"] += y_e
                    acc["pitch"] += p_e
                    acc["roll"] += r_e
                    acc["tx"] += tx_e
                    acc["ty"] += ty_e
                    acc["tz"] += tz_e
                    acc["l2"] += l2_e
                    acc["count"] += 1

                    _, R_ext, t_ext = get_cal(subj)
                    R_biwi = np.matmul(R_ext.T, R_pred_rgb)
                    t_biwi_mm = np.matmul(R_ext.T, (t_pred_rgb_m * 1000.0 - t_ext))
                    yb, pb, rb = base.biwi_euler_deg_batch(torch.from_numpy(R_biwi).float().unsqueeze(0))
                    per_frame[(subj, int(frame_num))] = {
                        "euler": (float(yb[0].item()), float(pb[0].item()), float(rb[0].item())),
                        "R_biwi": R_biwi.astype(np.float32),
                        "t_biwi": t_biwi_mm.astype(np.float32),
                    }

        count = int(sums["count"])
        if count == 0:
            raise RuntimeError("TRG native evaluation produced 0 samples.")
        overall = {
            "count": count,
            "yaw": sums["yaw"] / count,
            "pitch": sums["pitch"] / count,
            "roll": sums["roll"] / count,
            "mean": (sums["yaw"] + sums["pitch"] + sums["roll"]) / (3.0 * count),
            "tx": sums["tx"] / count,
            "ty": sums["ty"] / count,
            "tz": sums["tz"] / count,
        }
        overall["l2"] = float(np.sqrt(overall["tx"] ** 2 + overall["ty"] ** 2 + overall["tz"] ** 2))

        per_subject = {}
        for subj, acc in per_subject_acc.items():
            c = int(acc["count"])
            if c == 0:
                continue
            per_subject[subj] = {
                "count": c,
                "yaw": acc["yaw"] / c,
                "pitch": acc["pitch"] / c,
                "roll": acc["roll"] / c,
                "mean": (acc["yaw"] + acc["pitch"] + acc["roll"]) / (3.0 * c),
                "tx": acc["tx"] / c,
                "ty": acc["ty"] / c,
                "tz": acc["tz"] / c,
                "l2": acc["l2"] / c,
            }
        return {
            "overall": overall,
            "per_subject": per_subject,
            "per_frame": per_frame,
        }
    finally:
        model_zoo.load_url = old_load_url
        os.chdir(old_cwd)
        sys.path = old_sys_path

        for name in list(sys.modules.keys()):
            if any(name == p or name.startswith(f"{p}.") for p in prefixes):
                sys.modules.pop(name, None)
        sys.modules.update(old_modules)
        gc.collect()
        torch.cuda.empty_cache()


def evaluate_trg_native(args):
    """
    Native TRG BIWI evaluation protocol (native dataset/annotation/preprocess).
    """
    if args.no_trg:
        return None

    repo_root = os.path.abspath(args.trg_repo)
    if not os.path.isdir(repo_root):
        raise FileNotFoundError(f"TRG repo not found: {repo_root}")

    if not torch.cuda.is_available():
        raise RuntimeError("TRG native protocol requires CUDA (official test.py uses device='cuda').")

    ckpt_path = _resolve_trg_checkpoint_path(args)
    annot_path = _resolve_trg_native_annot_path(args, repo_root)
    layout_state = _prepare_trg_native_dataset_layout(args, repo_root, annot_path)

    logging.info("Running native TRG evaluation (native dataset + model pipeline)")
    try:
        out = _run_trg_native_detailed_inprocess(args, repo_root, ckpt_path)
    finally:
        _restore_trg_native_dataset_layout(layout_state)
    overall = out["overall"]

    logging.info(
        "[TRG native] overall — "
        f"Yaw: {overall['yaw']:.4f}  Pitch: {overall['pitch']:.4f}  Roll: {overall['roll']:.4f}  "
        f"MAE: {overall['mean']:.4f}  Tx: {overall['tx']:.4f}  Ty: {overall['ty']:.4f}  "
        f"Tz: {overall['tz']:.4f}  L2: {overall['l2']:.4f}"
    )
    return out


def _sixdof_face_root_is_ready(path):
    path = Path(path)
    return (
        (path / "csv" / "metadata_biwi.csv").is_file()
        and (path / "image").is_dir()
        and (path / "info").is_dir()
    )


def _prepare_sixdof_face_data_root_if_needed(args):
    data_root = _resolve_sixdof_face_data_root(args)
    if _sixdof_face_root_is_ready(data_root):
        return data_root

    if args.no_sixdof_face:
        return data_root

    if not args.prepare_sixdof_face_if_missing:
        return data_root

    logging.info(
        "6DoF Face BIWI evaluation root is missing; preparing it from raw BIWI data at %s",
        args.biwi_dir,
    )
    subjects = [str(s).zfill(2) for s in args.test_subjects] if args.test_subjects else None
    prepare_biwi_for_sixdof_face(
        biwi_root=args.biwi_dir,
        output_root=data_root,
        subjects=subjects,
        overwrite=False,
    )
    return data_root


def _validate_sixdof_face_paths(args):
    if args.no_sixdof_face:
        return

    if not os.path.isdir(args.sixdof_face_repo):
        raise FileNotFoundError(f"6DoF Face repo not found: {args.sixdof_face_repo}")

    checkpoints_dir = args.sixdof_face_checkpoints_dir or os.path.join(args.sixdof_face_repo, "checkpoint")
    ckpt_path = os.path.join(checkpoints_dir, args.sixdof_face_name, f"{args.sixdof_face_epoch}_net_R.pth")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"6DoF Face checkpoint not found: {ckpt_path}")

    data_root = _prepare_sixdof_face_data_root_if_needed(args)
    if not _sixdof_face_root_is_ready(data_root):
        raise FileNotFoundError(
            f"6DoF Face BIWI evaluation root not found: {data_root}\n"
            f"Pass --sixdof_face_data_root to the preprocessed BIWI eval root used by 6DoF Face "
            f"(it should contain csv/, image/, and info/), or enable --prepare_sixdof_face_if_missing."
        )


def evaluate_sixdof_face(args):
    """
    Run the native 6DoF Face BIWI evaluation pipeline and return overall/per-subject metrics.

    The 6DoF code reports translation errors in meters internally, so we convert tx/ty/tz/L2 to mm.
    """
    _validate_sixdof_face_paths(args)
    if args.no_sixdof_face:
        return None, {}

    repo_root = os.path.abspath(args.sixdof_face_repo)
    data_root = os.path.abspath(_prepare_sixdof_face_data_root_if_needed(args))
    checkpoints_dir = os.path.abspath(
        args.sixdof_face_checkpoints_dir or os.path.join(args.sixdof_face_repo, "checkpoint")
    )

    old_cwd = os.getcwd()
    old_sys_path = list(sys.path)

    orig_solvepnp_ransac = None

    try:
        os.chdir(repo_root)
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)

        from options.test_options import TestOptions
        from models import create_model
        if "data.augmentation" not in sys.modules:
            augmentation_stub = types.ModuleType("data.augmentation")

            class _UnusedAugmentor:
                def __init__(self, *args, **kwargs):
                    pass

            augmentation_stub.EulerAugmentor = _UnusedAugmentor
            augmentation_stub.HorizontalFlipAugmentor = _UnusedAugmentor
            sys.modules["data.augmentation"] = augmentation_stub
        import models.model_repository as model_repository_module

        orig_resnet18 = model_repository_module.resnet18

        def _offline_resnet18(*args, **kwargs):
            kwargs["pretrained"] = False
            return orig_resnet18(*args, **kwargs)

        model_repository_module.resnet18 = _offline_resnet18
        import data.biwi_dataset as biwi_dataset_module

        biwi_dataset_module.data_root = Path(data_root)
        BIWIDataset = _build_patched_biwi_dataset_class(biwi_dataset_module)
        orig_solvepnp_ransac = cv2.solvePnPRansac
        cv2.solvePnPRansac = _build_safe_solvepnp_ransac(orig_solvepnp_ransac)

        gpu_ids = str(args.gpu if torch.cuda.is_available() else -1)
        cmd_line = " ".join(
            [
                "--dataset_mode", "biwi",
                "--model", args.sixdof_face_model,
                "--img_size", str(args.sixdof_face_img_size),
                "--batch_size", str(args.sixdof_face_batch_size),
                "--num_threads", str(args.sixdof_face_num_workers),
                "--checkpoints_dir", checkpoints_dir,
                "--name", args.sixdof_face_name,
                "--epoch", args.sixdof_face_epoch,
                "--gpu_ids", gpu_ids,
                "--use_gt_bbox",
            ]
        )

        opt = TestOptions(cmd_line=cmd_line).parse()
        dataset = BIWIDataset(opt)
        if args.test_subjects:
            subject_ids = set(int(s) for s in args.test_subjects)
            dataset.df = dataset.df[dataset.df["subject_id"].astype(int).isin(subject_ids)].reset_index(drop=True)

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=int(opt.num_threads),
            drop_last=False,
        )

        model = create_model(opt)
        per_subject_acc = defaultdict(_empty_sixdof_acc)
        did_init = False

        for batch_idx, data in enumerate(dataloader):
            if batch_idx % 20 == 0:
                logging.info(f"[6DoF Face] batch {batch_idx + 1}/{len(dataloader)}")

            if not did_init:
                model.data_dependent_initialize(data)
                model.setup(opt)
                model.parallelize()
                model.eval()
                model.init_evaluation()
                did_init = True

            model.set_input(data)
            start_idx = len(model.inference_data["yaw_mae"])
            try:
                model.inference_curr_batch()
            except cv2.error as e:
                logging.warning(f"[6DoF Face] batch {batch_idx + 1}: OpenCV error, skipping batch — {e}")
                continue
            end_idx = len(model.inference_data["yaw_mae"])
            new_count = end_idx - start_idx

            batch_paths = list(model.img_path_list)[:new_count]
            for local_idx in range(new_count):
                subj = _parse_sixdof_subject(batch_paths[local_idx])
                yaw = float(model.inference_data["yaw_mae"][start_idx + local_idx])
                pitch = float(model.inference_data["pitch_mae"][start_idx + local_idx])
                roll = float(model.inference_data["roll_mae"][start_idx + local_idx])
                tx_m = float(model.inference_data["tx_mae"][start_idx + local_idx])
                ty_m = float(model.inference_data["ty_mae"][start_idx + local_idx])
                tz_m = float(model.inference_data["tz_mae"][start_idx + local_idx])

                acc = per_subject_acc[subj]
                acc["yaw"] += yaw
                acc["pitch"] += pitch
                acc["roll"] += roll
                acc["tx"] += tx_m * 1000.0
                acc["ty"] += ty_m * 1000.0
                acc["tz"] += tz_m * 1000.0
                acc["l2"] += float(np.sqrt(tx_m ** 2 + ty_m ** 2 + tz_m ** 2) * 1000.0)
                acc["count"] += 1

        yaw_arr = np.asarray(model.inference_data["yaw_mae"], dtype=np.float64)
        pitch_arr = np.asarray(model.inference_data["pitch_mae"], dtype=np.float64)
        roll_arr = np.asarray(model.inference_data["roll_mae"], dtype=np.float64)
        tx_arr_m = np.asarray(model.inference_data["tx_mae"], dtype=np.float64)
        ty_arr_m = np.asarray(model.inference_data["ty_mae"], dtype=np.float64)
        tz_arr_m = np.asarray(model.inference_data["tz_mae"], dtype=np.float64)

        overall = {
            "yaw": float(yaw_arr.mean()) if yaw_arr.size else float("nan"),
            "pitch": float(pitch_arr.mean()) if pitch_arr.size else float("nan"),
            "roll": float(roll_arr.mean()) if roll_arr.size else float("nan"),
            "mean": float((yaw_arr.mean() + pitch_arr.mean() + roll_arr.mean()) / 3.0) if yaw_arr.size else float("nan"),
            "tx": float(tx_arr_m.mean() * 1000.0) if tx_arr_m.size else float("nan"),
            "ty": float(ty_arr_m.mean() * 1000.0) if ty_arr_m.size else float("nan"),
            "tz": float(tz_arr_m.mean() * 1000.0) if tz_arr_m.size else float("nan"),
            "l2": float(np.sqrt(tx_arr_m ** 2 + ty_arr_m ** 2 + tz_arr_m ** 2).mean() * 1000.0) if tx_arr_m.size else float("nan"),
            "count": int(yaw_arr.size),
        }

        per_subject = {}
        for subj, acc in per_subject_acc.items():
            count = acc["count"]
            if count == 0:
                continue
            per_subject[subj] = {
                "count": count,
                "yaw": acc["yaw"] / count,
                "pitch": acc["pitch"] / count,
                "roll": acc["roll"] / count,
                "mean": (acc["yaw"] + acc["pitch"] + acc["roll"]) / (3 * count),
                "tx": acc["tx"] / count,
                "ty": acc["ty"] / count,
                "tz": acc["tz"] / count,
                "l2": acc["l2"] / count,
            }

        logging.info(
            "[6DoF Face] overall — "
            f"Yaw: {overall['yaw']:.4f}  Pitch: {overall['pitch']:.4f}  Roll: {overall['roll']:.4f}  "
            f"MAE: {overall['mean']:.4f}  Tx: {overall['tx']:.4f}  Ty: {overall['ty']:.4f}  "
            f"Tz: {overall['tz']:.4f}  L2: {overall['l2']:.4f}"
        )

        return overall, per_subject

    finally:
        if orig_solvepnp_ransac is not None:
            cv2.solvePnPRansac = orig_solvepnp_ransac
        os.chdir(old_cwd)
        sys.path = old_sys_path
        gc.collect()
        torch.cuda.empty_cache()


def evaluate(args):
    if args.min_anchor_frame_gap < 0:
        raise ValueError("--min_anchor_frame_gap must be >= 0")
    if args.close_anchor_min_frame_gap < 0:
        raise ValueError("--close_anchor_min_frame_gap must be >= 0")
    if args.max_anchor_pose_gap_deg <= 0:
        raise ValueError("--max_anchor_pose_gap_deg must be > 0")

    _validate_sixdof_face_paths(args)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    amp_dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
        else torch.float16
    )
    variant_names = base.get_variant_names(args.vggt_pose_mode)
    primary_variant = base.get_primary_variant_name(args.vggt_pose_mode)

    test_subjects = set(args.test_subjects)
    logging.info(f"Device: {device}  |  AMP: {amp_dtype}")
    logging.info(f"Test subjects: {sorted(test_subjects)}")
    logging.info(f"VGGT pose mode: {args.vggt_pose_mode}")
    logging.info("VGGT crop mode: square MTCNN crop")
    logging.info("6DRepNet crop mode: rectangular MTCNN crop (matches training distortion)")
    logging.info("TokenHPE crop mode: rectangular MTCNN crop -> resize 250 -> center-crop 224")
    logging.info("WHENet crop mode: rectangular MTCNN crop -> resize 224")
    if args.trg_shared_crop_source == "trg_fan":
        logging.info("TRG crop mode: TRG FAN-keypoint bbox (native-style) + square affine crop + matched K_crop/bbox_info")
    else:
        logging.info("TRG crop mode: square affine crop from MTCNN box (TRG-style), with matched K_crop/bbox_info")
    if args.native_protocol:
        logging.info("Native protocol mode enabled for TokenHPE/TRG (official eval pipelines).")
        logging.info("Native TokenHPE/TRG predictions will be merged per-frame into shared subject reports/plots when frame mapping exists.")
        if set(args.test_subjects) != set(range(1, 25)):
            logging.warning(
                "native_protocol note: TokenHPE/TRG native evaluators do not honor --test_subjects filtering "
                "(they follow each method's native test split/protocol)."
            )
    else:
        logging.warning(
            "TokenHPE protocol note: current compare uses shared online MTCNN crops from raw BIWI frames; "
            "official TokenHPE BIWI eval uses preprocessed BIWI npz crops/split."
        )
        logging.warning(
            "TRG protocol note: current compare uses shared MTCNN-derived boxes; "
            "official TRG BIWI eval uses annot_mtcnn_fan.pkl (FAN landmarks) and native dataset pipeline."
        )
        logging.warning(
            "WHENet protocol note: current compare uses shared online MTCNN crops from raw BIWI frames; "
            "official WHENet repo provides demo inference (no native BIWI benchmark script)."
        )
    logging.info(f"VGGT translation scale: x{args.vggt_translation_scale:.4f}")
    logging.info("VGGT translation is decoded in BIWI pose.txt space (depth-camera frame, mm).")
    if args.vggt_pose_mode in {"relative_h2c", "absolute_second"}:
        logging.info(f"Anchor selection mode: {args.anchor_selection}")
        if args.anchor_selection in {"random_far", "uniform_gap", "close_pose"}:
            seed_msg = "random" if args.anchor_seed < 0 else str(args.anchor_seed)
            if args.anchor_selection in {"random_far", "uniform_gap"}:
                logging.info(
                    f"Random anchor constraint: |Δframe| >= {args.min_anchor_frame_gap}  (seed={seed_msg})"
                )
            else:
                logging.info(
                    f"Close-pose anchor constraint: |Δframe| >= {args.close_anchor_min_frame_gap}, "
                    f"ΔR <= {args.max_anchor_pose_gap_deg:.2f} deg  (seed={seed_msg})"
                )
            if args.anchor_selection == "uniform_gap":
                logging.info("uniform_gap additionally balances anchor-target rotation-gap histogram bins.")
        elif args.anchor_selection == "autoregressive":
            logging.info("Autoregressive anchor mode: anchor pose comes from latest VGGT prediction.")
            logging.info("Autoregressive chain starts from frame 0 GT anchor and proceeds sequentially.")
    elif args.anchor_selection != "fixed_first":
        logging.warning(
            "anchor_selection is ignored for absolute_single mode (single-view input)."
        )
    anchor_gap_hist = None
    if args.vggt_pose_mode in {"relative_h2c", "absolute_second"}:
        anchor_gap_hist = _init_anchor_gap_hist(args.anchor_gap_bins_deg)
        logging.info(
            "Anchor-gap histogram bins (deg): %s",
            ", ".join(f"{v:.1f}" for v in anchor_gap_hist["edges"]),
        )
    else:
        logging.info("Anchor-gap histogram disabled for absolute_single mode.")
    os.makedirs(args.vis_dir, exist_ok=True)

    sixdof_overall, sixdof_per_subject = None, {}
    tokenhpe_native_overall = None
    tokenhpe_native_by_frame = {}
    trg_native_overall = None
    trg_native_by_frame = {}
    trg_native_per_subject = {}
    try:
        logging.info("Running 6DoF Face pre-pass so per-subject lines can include all methods.")
        sixdof_overall, sixdof_per_subject = evaluate_sixdof_face(args)
    except Exception as e:
        logging.error(f"6DoF Face evaluation failed: {e}")
        logging.error("Continuing with VGGT/6DRepNet/TokenHPE/WHENet/TRG results only.")

    if args.native_protocol:
        if not args.no_tokenhpe:
            token_native = evaluate_tokenhpe_native(args, device)
            tokenhpe_native_overall = token_native["overall"]
            tokenhpe_native_by_frame = token_native["per_frame"]
        if not args.no_trg:
            trg_native = evaluate_trg_native(args)
            trg_native_overall = trg_native["overall"]
            trg_native_by_frame = trg_native["per_frame"]
            trg_native_per_subject = trg_native.get("per_subject", {})
        if args.native_protocol_keep_shared:
            logging.info(
                "native_protocol active: native evaluators enabled AND shared TokenHPE/TRG inference kept."
            )
        else:
            logging.info(
                "native_protocol active: shared TokenHPE/TRG inference is disabled; "
                "per-subject TokenHPE/TRG lines are native-merged on current subject frames."
            )

    lora_path = None if args.no_lora else args.lora_checkpoint
    vggt_model = base.load_vggt_model(
        args.base_checkpoint,
        lora_path,
        device,
        enable_track=args.enable_track,
    )

    sixd_model = None
    if not args.no_sixdrepnet:
        sixd_model = base.load_sixdrepnet_model(
            args.sixdrepnet_checkpoint,
            args.sixdrepnet_dir,
            device,
        )
    tokenhpe_model = None
    if (not args.no_tokenhpe) and ((not args.native_protocol) or args.native_protocol_keep_shared):
        try:
            tokenhpe_model = load_tokenhpe_model(args, device)
        except Exception as e:
            logging.error(f"TokenHPE initialization failed: {e}")
            logging.error("Continuing with VGGT/6DRepNet/6DoF Face/WHENet/TRG results only.")
    whenet_model = None
    if not args.no_whenet:
        try:
            whenet_model = load_whenet_model(args)
        except Exception as e:
            logging.error(f"WHENet initialization failed: {e}")
            logging.error("Continuing with VGGT/6DRepNet/TokenHPE/6DoF Face/TRG results only.")
    trg_model = None
    if (not args.no_trg) and ((not args.native_protocol) or args.native_protocol_keep_shared):
        try:
            trg_model = load_trg_model(args, device)
        except Exception:
            logging.exception("TRG initialization failed; aborting evaluation.")
            raise
    trg_fan_pred_kpt_map = None
    trg_fan_bbox_hit = 0
    trg_fan_bbox_fallback = 0
    if (trg_model is not None) and (args.trg_shared_crop_source == "trg_fan"):
        trg_fan_pred_kpt_map, trg_annot_path = _load_trg_fan_keypoints_map(args)
        logging.info(
            "TRG shared FAN bbox source loaded: %s (pred_kpt entries: %d)",
            trg_annot_path,
            len(trg_fan_pred_kpt_map),
        )

    intersection_enabled = False
    intersection_methods = []
    intersection_global_count = 0
    intersection_global_sums = {}
    if args.native_protocol:
        intersection_methods = ["vggt"]
        if sixd_model is not None:
            intersection_methods.append("sixdrepnet")
        if (tokenhpe_model is not None) or bool(tokenhpe_native_by_frame):
            intersection_methods.append("tokenhpe")
        if whenet_model is not None:
            intersection_methods.append("whenet")
        if (trg_model is not None) or bool(trg_native_by_frame):
            intersection_methods.append("trg")
        if len(intersection_methods) >= 2:
            intersection_enabled = True
            intersection_global_sums = {
                m: {"yaw": 0.0, "pitch": 0.0, "roll": 0.0} for m in intersection_methods
            }
            logging.info(
                "Intersection metrics enabled (native_protocol): %s",
                ", ".join(_method_display_name(m) for m in intersection_methods),
            )
        else:
            logging.info("Intersection metrics skipped: fewer than two active methods.")

    mtcnn_detector = load_shared_face_detector(args, device)
    logging.info(
        "Shared face detector initialized: %s (crop expansion %.3f)",
        args.face_detector,
        args.mtcnn_ad,
    )

    mesh_cache, cal_cache = {}, {}

    def get_mesh(subj):
        if subj not in mesh_cache:
            mesh_cache[subj] = base.load_obj(os.path.join(args.biwi_dir, f"{subj}.obj"))
        return mesh_cache[subj]

    def get_cal(subj):
        if subj not in cal_cache:
            cal_cache[subj] = base.load_rgb_intrinsics(os.path.join(args.biwi_dir, subj, "rgb.cal"))
        return cal_cache[subj]

    all_dirs = sorted(glob.glob(os.path.join(args.biwi_dir, "*")))
    test_dirs = [
        d
        for d in all_dirs
        if os.path.isdir(d)
        and os.path.basename(d).isdigit()
        and int(os.path.basename(d)) in test_subjects
    ]
    pair_subset_by_subject = _load_pair_subset_csv(args.pair_csv) if args.pair_csv else None
    logging.info(f"Evaluating {len(test_dirs)} VGGT test subjects\n")

    vggt_tot_y = vggt_tot_p = vggt_tot_r = 0.0
    sixd_tot_y = sixd_tot_p = sixd_tot_r = 0.0
    tokenhpe_tot_y = tokenhpe_tot_p = tokenhpe_tot_r = 0.0
    whenet_tot_y = whenet_tot_p = whenet_tot_r = 0.0
    trg_tot_y = trg_tot_p = trg_tot_r = 0.0
    trg_tot_t = vggt_translation._empty_translation_acc()
    tokenhpe_eval_count = 0
    whenet_eval_count = 0
    trg_eval_count = 0
    vggt_tot_t = vggt_translation._empty_translation_acc()
    total_count = 0
    vis_count = 0

    per_subject = {}
    pair_records_all = []
    global_variant_errors = base._empty_variant_acc(variant_names)

    for subj_dir in test_dirs:
        subj = os.path.basename(subj_dir)

        if not os.path.exists(os.path.join(args.biwi_dir, f"{subj}.obj")):
            logging.warning(f"Subject {subj}: no .obj mesh, skipping.")
            continue
        if not os.path.exists(os.path.join(subj_dir, "rgb.cal")):
            logging.warning(f"Subject {subj}: no rgb.cal, skipping.")
            continue

        frames = base.get_subject_frames(subj_dir)
        use_anchor_pair = args.vggt_pose_mode in {"relative_h2c", "absolute_second"}
        min_required_frames = 2 if use_anchor_pair else 1
        if len(frames) < min_required_frames:
            logging.warning(f"Subject {subj}: <{min_required_frames} frames, skipping.")
            continue

        subj_pair_subset = None
        if pair_subset_by_subject is not None:
            subj_pair_subset = pair_subset_by_subject.get(subj, [])
            if len(subj_pair_subset) == 0:
                logging.info(f"Subject {subj}: no rows in pair CSV, skipping.")
                continue

        vertices, faces = get_mesh(subj)
        K_rgb, R_cam, t_cam = get_cal(subj)
        R_cam_t = torch.from_numpy(R_cam).float()
        t_cam_t = torch.from_numpy(t_cam).float()

        anchor_tensor = None
        R_anchor_t = None
        t_anchor_t = None
        fixed_anchor_frame_num = None
        fixed_anchor_img_rgb = None
        ar_anchor_tensor = None
        ar_anchor_R_pred = None
        ar_anchor_t_pred = None
        ar_anchor_R_gt = None
        ar_anchor_t_gt = None
        ar_anchor_frame_num = None
        ar_anchor_img_rgb = None
        random_anchor_candidates = {}
        subject_rng = None
        uniform_edges = None
        uniform_subject_counts = None
        uniform_global_counts = None
        pose_cache = {}
        anchor_cache = {}

        def get_frame_pose(idx):
            if idx not in pose_cache:
                _, _, pose_path = frames[idx]
                R_np, t_np = base.parse_pose_txt(pose_path)
                pose_cache[idx] = (
                    torch.from_numpy(R_np).float(),
                    torch.from_numpy(t_np).float(),
                )
            return pose_cache[idx]

        def get_anchor_data(anchor_idx):
            if anchor_idx in anchor_cache:
                return anchor_cache[anchor_idx]

            anchor_frame_num, anchor_rgb_path, _ = frames[anchor_idx]
            R_anchor_torch, t_anchor_torch = get_frame_pose(anchor_idx)
            anchor_bgr = cv2.imread(anchor_rgb_path)
            if anchor_bgr is None:
                logging.warning(
                    f"Subject {subj}: failed to read anchor frame {anchor_rgb_path}, dropping this anchor."
                )
                anchor_cache[anchor_idx] = None
                return None

            # Anchor is only used by VGGT, so square crop is fine here.
            anchor_crop_rgb, _, _ = square_crop.crop_face_mtcnn_square_for_vggt(
                anchor_bgr,
                mtcnn_detector,
                None,
                crop_size=args.crop_size,
                ad=args.mtcnn_ad,
            )
            anchor_img_rgb = cv2.cvtColor(anchor_bgr, cv2.COLOR_BGR2RGB) if args.visualize else None
            anchor_data = (
                base.vggt_transform(anchor_crop_rgb),
                R_anchor_torch,
                t_anchor_torch,
                anchor_frame_num,
                anchor_img_rgb,
            )
            anchor_cache[anchor_idx] = anchor_data
            return anchor_data

        if args.pair_csv:
            eval_groups = _build_subject_eval_groups_from_pair_csv(
                subj,
                frames,
                use_anchor_pair,
                subj_pair_subset,
            )
            if len(eval_groups) == 0:
                logging.warning(f"Subject {subj}: no valid pair-CSV evaluation groups, skipping.")
                continue

            total_targets = sum(len(group["targets"]) for group in eval_groups)
            logging.info(f"[{subj}] pair-CSV mode: {len(eval_groups)} anchor groups, {total_targets} target pairs")

            s_variant_errors = base._empty_variant_acc(variant_names)
            s_sixd_y = s_sixd_p = s_sixd_r = 0.0
            s_tokenhpe_y = s_tokenhpe_p = s_tokenhpe_r = 0.0
            s_whenet_y = s_whenet_p = s_whenet_r = 0.0
            s_trg_y = s_trg_p = s_trg_r = 0.0
            s_tokenhpe_count = 0
            s_whenet_count = 0
            s_trg_count = 0
            s_trg_t = vggt_translation._empty_translation_acc()
            s_vggt_t = vggt_translation._empty_translation_acc()
            s_count = 0

            for group_idx, group in enumerate(eval_groups):
                anchor_tensor_group = None
                R_anchor_group = None
                t_anchor_group = None
                mtcnn_prev_box = None

                if use_anchor_pair:
                    anchor_frame_num, _, _ = group["anchor"]
                    anchor_idx = next((i for i, fr in enumerate(frames) if fr[0] == anchor_frame_num), None)
                    if anchor_idx is None:
                        logging.warning(f"Subject {subj}: failed to map anchor frame {anchor_frame_num}, skipping group.")
                        continue
                    anchor_data = get_anchor_data(anchor_idx)
                    if anchor_data is None:
                        logging.warning(f"Subject {subj}: failed to load anchor frame {anchor_frame_num}, skipping group.")
                        continue
                    (
                        anchor_tensor_group,
                        R_anchor_group,
                        t_anchor_group,
                        _,
                        anchor_img_rgb_group,
                    ) = anchor_data

                target_frames = group["targets"]
                n_targets = len(target_frames)
                n_batches = (n_targets + args.batch_size - 1) // args.batch_size
                if use_anchor_pair:
                    logging.info(
                        f"  group {group_idx + 1}/{len(eval_groups)}: anchor=frame {group['anchor'][0]}, "
                        f"{n_targets} targets, {n_batches} batches"
                    )
                else:
                    logging.info(f"  group {group_idx + 1}/{len(eval_groups)}: {n_targets} targets, {n_batches} batches")

                for b_idx, b_start in enumerate(range(0, n_targets, args.batch_size)):
                    batch = target_frames[b_start : b_start + args.batch_size]
                    if b_idx % 20 == 0:
                        logging.info(f"    batch {b_idx + 1}/{n_batches}")

                    vggt_tensors, sixd_tensors, tokenhpe_tensors, whenet_inputs, trg_tensors = [], [], [], [], []
                    trg_bbox_infos, trg_intrinsics = [], []
                    batch_anchor_R, batch_anchor_t = [], []
                    batch_anchor_frame_nums, batch_anchor_imgs_rgb = [], []
                    frame_nums, R_targets, t_targets = [], [], []
                    imgs_rgb = []

                    for frame_num, rgb_path, pose_path in batch:
                        R_tgt, t_tgt = base.parse_pose_txt(pose_path)
                        img_bgr = cv2.imread(rgb_path)
                        if img_bgr is None:
                            logging.warning(f"Failed to read target frame {rgb_path}, skipping frame.")
                            continue
                        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) if args.visualize else None

                        rect_crop_bgr, mtcnn_prev_box = base.crop_face_mtcnn(
                            img_bgr,
                            mtcnn_detector,
                            mtcnn_prev_box,
                            crop_size=args.crop_size,
                            ad=args.mtcnn_ad,
                        )
                        sq_crop_bgr = square_crop.square_crop_from_box(
                            img_bgr,
                            mtcnn_prev_box,
                            crop_size=args.crop_size,
                        )
                        sq_crop_rgb = cv2.cvtColor(sq_crop_bgr, cv2.COLOR_BGR2RGB)
                        vggt_tensors.append(base.vggt_transform(sq_crop_rgb))

                        if sixd_model is not None:
                            sixd_tensors.append(base.SIXD_TRANSFORM(Image.fromarray(rect_crop_bgr)))
                        if tokenhpe_model is not None:
                            tokenhpe_tensors.append(TOKENHPE_TRANSFORM(Image.fromarray(rect_crop_bgr)))
                        if whenet_model is not None:
                            rect_crop_rgb = cv2.cvtColor(rect_crop_bgr, cv2.COLOR_BGR2RGB)
                            whenet_inputs.append(cv2.resize(rect_crop_rgb, (224, 224), interpolation=cv2.INTER_LINEAR))
                        if trg_model is not None:
                            img_h, img_w = img_bgr.shape[:2]
                            left, top, right, bottom = _build_trg_square_bbox_from_mtcnn_box(
                                mtcnn_prev_box,
                                img_w,
                                img_h,
                            )
                            trg_crop_rgb = _trg_affine_crop(
                                img_bgr,
                                (left, top, right, bottom),
                                img_size=args.trg_img_size,
                            )
                            trg_tensors.append(TRG_TRANSFORM(Image.fromarray(trg_crop_rgb)))
                            bbox_info_i = np.array(
                                [float(left), float(top), float(right), float(bottom), float(K_rgb[0, 0]), float(img_h), float(img_w)],
                                dtype=np.float32,
                            )
                            K_crop_i = _build_trg_intrinsic_crop(K_rgb, bbox_info_i, args.trg_img_size)
                            trg_bbox_infos.append(torch.from_numpy(bbox_info_i).float())
                            trg_intrinsics.append(torch.from_numpy(K_crop_i).float())

                        frame_nums.append(frame_num)
                        R_targets.append(torch.from_numpy(R_tgt).float())
                        t_targets.append(torch.from_numpy(t_tgt).float())
                        imgs_rgb.append(img_rgb)
                        if use_anchor_pair:
                            batch_anchor_R.append(R_anchor_group)
                            batch_anchor_t.append(t_anchor_group)
                            batch_anchor_frame_nums.append(anchor_frame_num)
                            batch_anchor_imgs_rgb.append(anchor_img_rgb_group)

                    if len(vggt_tensors) == 0:
                        continue

                    bs = len(vggt_tensors)
                    R_gt_batch = torch.stack(R_targets)
                    t_gt_batch = torch.stack(t_targets)
                    y_gt, p_gt, r_gt = base.biwi_euler_deg_batch(R_gt_batch)
                    R_sixd_biwi = None
                    tokenhpe_pred_eulers = [None] * bs
                    tokenhpe_pred_R = [None] * bs
                    whenet_pred_eulers = [None] * bs
                    whenet_pred_R = [None] * bs
                    trg_pred_eulers = [None] * bs
                    trg_pred_R = [None] * bs

                    tgt_stack = torch.stack(vggt_tensors)
                    if use_anchor_pair:
                        anchor_rep = anchor_tensor_group.unsqueeze(0).expand(bs, -1, -1, -1)
                        vggt_images = torch.stack([anchor_rep, tgt_stack], dim=1).to(device)
                    else:
                        vggt_images = tgt_stack.unsqueeze(1).to(device)

                    with torch.no_grad(), torch.cuda.amp.autocast(dtype=amp_dtype):
                        vggt_out = vggt_model(images=vggt_images)

                    decode_out = vggt_translation.decode_vggt_predictions(
                        vggt_out["pose_enc"],
                        R_anchor_group,
                        t_anchor_group,
                        R_cam_t,
                        t_cam_t,
                        pose_mode=args.vggt_pose_mode,
                        translation_scale=args.vggt_translation_scale,
                        return_debug=args.visualize,
                    )
                    if args.visualize:
                        variants, t_vggt_primary, decode_debug = decode_out
                    else:
                        variants, t_vggt_primary = decode_out
                        decode_debug = None
                    if args.vggt_prediction_head_yz_flip:
                        variants = _apply_head_yz_flip_to_vggt_variants(variants)

                    for vname, R_v in variants.items():
                        yv, pv, rv = base.biwi_euler_deg_batch(R_v)
                        err_y = base.angular_error(y_gt, yv).sum().item()
                        err_p = base.angular_error(p_gt, pv).sum().item()
                        err_r = base.angular_error(r_gt, rv).sum().item()
                        global_variant_errors[vname]["y"] += err_y
                        global_variant_errors[vname]["p"] += err_p
                        global_variant_errors[vname]["r"] += err_r
                        s_variant_errors[vname]["y"] += err_y
                        s_variant_errors[vname]["p"] += err_p
                        s_variant_errors[vname]["r"] += err_r

                    R_vggt_primary = variants[primary_variant]
                    y_vp, p_vp, r_vp = base.biwi_euler_deg_batch(R_vggt_primary)
                    err_v_y = base.angular_error(y_gt, y_vp)
                    err_v_p = base.angular_error(p_gt, p_vp)
                    err_v_r = base.angular_error(r_gt, r_vp)
                    vggt_tot_y += err_v_y.sum().item()
                    vggt_tot_p += err_v_p.sum().item()
                    vggt_tot_r += err_v_r.sum().item()
                    vggt_translation.accumulate_translation_errors(vggt_tot_t, t_vggt_primary, t_gt_batch)
                    vggt_translation.accumulate_translation_errors(s_vggt_t, t_vggt_primary, t_gt_batch)

                    if sixd_model is not None:
                        sixd_imgs = torch.stack(sixd_tensors).to(device)
                        with torch.no_grad():
                            R_sixd_batch = sixd_model(sixd_imgs).cpu()
                        y_sp, p_sp, r_sp, R_sixd_biwi = base.decode_sixdrepnet_pose(R_sixd_batch)
                        err_sixd_y = base.angular_error(y_gt, y_sp)
                        err_sixd_p = base.angular_error(p_gt, p_sp)
                        err_sixd_r = base.angular_error(r_gt, r_sp)
                        s_sixd_y += err_sixd_y.sum().item()
                        s_sixd_p += err_sixd_p.sum().item()
                        s_sixd_r += err_sixd_r.sum().item()
                        sixd_tot_y += err_sixd_y.sum().item()
                        sixd_tot_p += err_sixd_p.sum().item()
                        sixd_tot_r += err_sixd_r.sum().item()

                    if tokenhpe_model is not None:
                        tokenhpe_imgs = torch.stack(tokenhpe_tensors).to(device)
                        with torch.no_grad():
                            R_tokenhpe_batch, _ = tokenhpe_model(tokenhpe_imgs)
                            R_tokenhpe_batch = R_tokenhpe_batch.cpu()
                        y_th, p_th, r_th, R_tokenhpe_biwi = base.decode_sixdrepnet_pose(R_tokenhpe_batch)
                        err_tokenhpe_y = base.angular_error(y_gt, y_th)
                        err_tokenhpe_p = base.angular_error(p_gt, p_th)
                        err_tokenhpe_r = base.angular_error(r_gt, r_th)
                        s_tokenhpe_y += err_tokenhpe_y.sum().item()
                        s_tokenhpe_p += err_tokenhpe_p.sum().item()
                        s_tokenhpe_r += err_tokenhpe_r.sum().item()
                        tokenhpe_tot_y += err_tokenhpe_y.sum().item()
                        tokenhpe_tot_p += err_tokenhpe_p.sum().item()
                        tokenhpe_tot_r += err_tokenhpe_r.sum().item()
                        s_tokenhpe_count += bs
                        tokenhpe_eval_count += bs
                        for fi in range(bs):
                            tokenhpe_pred_eulers[fi] = (y_th[fi].item(), p_th[fi].item(), r_th[fi].item())
                            tokenhpe_pred_R[fi] = R_tokenhpe_biwi[fi]

                    if whenet_model is not None:
                        whenet_batch = np.stack(whenet_inputs, axis=0).astype(np.float32)
                        y_wh_np, p_wh_np, r_wh_np = whenet_model.get_angle(whenet_batch)
                        y_wh = torch.from_numpy(np.asarray(y_wh_np, dtype=np.float32))
                        p_wh = torch.from_numpy(np.asarray(p_wh_np, dtype=np.float32))
                        r_wh = torch.from_numpy(np.asarray(r_wh_np, dtype=np.float32))
                        err_whenet_y = base.angular_error(y_gt, y_wh)
                        err_whenet_p = base.angular_error(p_gt, p_wh)
                        err_whenet_r = base.angular_error(r_gt, r_wh)
                        s_whenet_y += err_whenet_y.sum().item()
                        s_whenet_p += err_whenet_p.sum().item()
                        s_whenet_r += err_whenet_r.sum().item()
                        whenet_tot_y += err_whenet_y.sum().item()
                        whenet_tot_p += err_whenet_p.sum().item()
                        whenet_tot_r += err_whenet_r.sum().item()
                        s_whenet_count += bs
                        whenet_eval_count += bs
                        R_whenet_biwi = np.stack(
                            [
                                base.euler_to_R_biwi(
                                    float(y_wh_np[fi]),
                                    float(p_wh_np[fi]),
                                    float(r_wh_np[fi]),
                                )
                                for fi in range(bs)
                            ],
                            axis=0,
                        )
                        for fi in range(bs):
                            whenet_pred_eulers[fi] = (
                                float(y_wh_np[fi]),
                                float(p_wh_np[fi]),
                                float(r_wh_np[fi]),
                            )
                            whenet_pred_R[fi] = R_whenet_biwi[fi]

                    if trg_model is not None:
                        trg_imgs = torch.stack(trg_tensors).to(device)
                        trg_bbox = torch.stack(trg_bbox_infos).to(device)
                        trg_intr = torch.stack(trg_intrinsics).to(device)
                        with torch.no_grad():
                            preds_dict, _ = trg_model(trg_imgs, trg_intr, trg_bbox)
                            pred_R_t = preds_dict["output"][-1]["pred_R_t"].detach().cpu()

                        pred_M = pred_R_t.transpose(1, 2)
                        R_pred_rgb = pred_M[:, :3, :3]
                        t_pred_rgb_mm = pred_M[:, :3, 3] * 1000.0
                        R_ext_t = torch.from_numpy(R_cam).float()
                        t_ext_t = torch.from_numpy(t_cam).float()
                        R_trg_biwi = torch.matmul(R_ext_t.T.unsqueeze(0), R_pred_rgb)
                        t_trg_biwi = torch.matmul(
                            R_ext_t.T.unsqueeze(0),
                            (t_pred_rgb_mm - t_ext_t.unsqueeze(0)).unsqueeze(-1),
                        ).squeeze(-1)

                        y_trg, p_trg, r_trg = base.biwi_euler_deg_batch(R_trg_biwi)
                        err_trg_y = base.angular_error(y_gt, y_trg)
                        err_trg_p = base.angular_error(p_gt, p_trg)
                        err_trg_r = base.angular_error(r_gt, r_trg)
                        s_trg_y += err_trg_y.sum().item()
                        s_trg_p += err_trg_p.sum().item()
                        s_trg_r += err_trg_r.sum().item()
                        trg_tot_y += err_trg_y.sum().item()
                        trg_tot_p += err_trg_p.sum().item()
                        trg_tot_r += err_trg_r.sum().item()
                        vggt_translation.accumulate_translation_errors(s_trg_t, t_trg_biwi, t_gt_batch)
                        vggt_translation.accumulate_translation_errors(trg_tot_t, t_trg_biwi, t_gt_batch)
                        s_trg_count += bs
                        trg_eval_count += bs
                        for fi in range(bs):
                            trg_pred_eulers[fi] = (y_trg[fi].item(), p_trg[fi].item(), r_trg[fi].item())
                            trg_pred_R[fi] = R_trg_biwi[fi].numpy()

                    if args.visualize:
                        for fi in range(bs):
                            if vis_count % args.vis_every_n == 0:
                                frame_num = frame_nums[fi]
                                save_path = os.path.join(args.vis_dir, f"subj{subj}_frame{frame_num:05d}.png")

                                euler_gt_i = (y_gt[fi].item(), p_gt[fi].item(), r_gt[fi].item())
                                vggt_vis = {}
                                for vname, R_v in variants.items():
                                    yv, pv, rv = base.biwi_euler_deg_batch(R_v)
                                    vggt_vis[vname] = (
                                        R_v[fi].numpy(),
                                        (yv[fi].item(), pv[fi].item(), rv[fi].item()),
                                    )

                                euler_sixd_i = (
                                    (y_sp[fi].item(), p_sp[fi].item(), r_sp[fi].item())
                                    if y_sp is not None
                                    else None
                                )
                                euler_tokenhpe_i = tokenhpe_pred_eulers[fi]
                                euler_whenet_i = whenet_pred_eulers[fi]
                                euler_trg_i = trg_pred_eulers[fi]
                                prefix_panels = []
                                anchor_y = anchor_p = anchor_r = None
                                if use_anchor_pair:
                                    anchor_y, anchor_p, anchor_r = _euler_from_rotation(batch_anchor_R[fi])

                                if use_anchor_pair and batch_anchor_imgs_rgb[fi] is not None:
                                    anchor_panel = base.render_mesh_on_image(
                                        batch_anchor_imgs_rgb[fi].copy(),
                                        vertices,
                                        faces,
                                        batch_anchor_R[fi].numpy(),
                                        batch_anchor_t[fi].numpy(),
                                        R_cam,
                                        t_cam,
                                        K_rgb,
                                        color="#4a90d9",
                                        alpha=0.5,
                                    )
                                    prefix_panels.append(
                                        (
                                            anchor_panel,
                                            f"Anchor GT f{batch_anchor_frame_nums[fi]}\nY:{anchor_y:.1f}  P:{anchor_p:.1f}  R:{anchor_r:.1f}",
                                        )
                                    )

                                if (
                                    use_anchor_pair
                                    and decode_debug is not None
                                    and args.vggt_pose_mode == "relative_h2c"
                                    and "R_rel_depth" in decode_debug
                                ):
                                    rel_y, rel_p, rel_r = _euler_from_rotation(decode_debug["R_rel_depth"][fi])
                                    final_y, final_p, final_r = _euler_from_rotation(decode_debug["R_final_primary"][fi])
                                    decomp_panel = _make_text_panel_like(
                                        imgs_rgb[fi],
                                        [
                                            "VGGT Relative Decomposition",
                                            f"R_rel (depth/OpenCV): Y:{rel_y:.1f} P:{rel_p:.1f} R:{rel_r:.1f}",
                                            f"R_anchor (used):       Y:{anchor_y:.1f} P:{anchor_p:.1f} R:{anchor_r:.1f}",
                                            "R_final = R_rel @ R_anchor",
                                            f"R_final ({primary_variant}): Y:{final_y:.1f} P:{final_p:.1f} R:{final_r:.1f}",
                                        ],
                                    )
                                    prefix_panels.append((decomp_panel, "Relative -> Final (same frame)"))

                                if len(prefix_panels) == 0:
                                    prefix_panels = None

                                base.save_vis_frame(
                                    img_rgb=imgs_rgb[fi],
                                    vertices=vertices,
                                    faces=faces,
                                    K_rgb=K_rgb,
                                    R_cam=R_cam,
                                    t_cam=t_cam,
                                    R_gt=R_gt_batch[fi].numpy(),
                                    t_gt=t_gt_batch[fi].numpy(),
                                    euler_gt=euler_gt_i,
                                    vggt_variants=vggt_vis,
                                    R_sixd=R_sixd_biwi[fi] if R_sixd_biwi is not None else None,
                                    euler_sixd=euler_sixd_i,
                                    R_tokenhpe=tokenhpe_pred_R[fi],
                                    euler_tokenhpe=euler_tokenhpe_i,
                                    R_whenet=whenet_pred_R[fi],
                                    euler_whenet=euler_whenet_i,
                                    R_trg=trg_pred_R[fi],
                                    euler_trg=euler_trg_i,
                                    save_path=save_path,
                                    prefix_panels=prefix_panels,
                                )
                            vis_count += 1

                    s_count += bs

            if s_count > 0:
                per_subject[subj] = {}
                for vname in variant_names:
                    vy = s_variant_errors[vname]["y"] / s_count
                    vp = s_variant_errors[vname]["p"] / s_count
                    vr = s_variant_errors[vname]["r"] / s_count
                    vm = (vy + vp + vr) / 3.0
                    per_subject[subj][f"vggt_{vname}"] = {
                        "yaw": vy,
                        "pitch": vp,
                        "roll": vr,
                        "mean": vm,
                    }

                t_metrics = vggt_translation.translation_metrics_from_acc(s_vggt_t, s_count)
                per_subject[subj][f"vggt_{primary_variant}"].update(t_metrics)

                line = (
                    f"Subject {subj} ({s_count} frames):\n"
                    f"  VGGT ({primary_variant}) — Yaw: {per_subject[subj][f'vggt_{primary_variant}']['yaw']:.2f}  "
                    f"Pitch: {per_subject[subj][f'vggt_{primary_variant}']['pitch']:.2f}  "
                    f"Roll: {per_subject[subj][f'vggt_{primary_variant}']['roll']:.2f}  "
                    f"MAE: {per_subject[subj][f'vggt_{primary_variant}']['mean']:.2f}\n"
                    f"  VGGT Translation    — Tx: {t_metrics['tx']:.2f}  Ty: {t_metrics['ty']:.2f}"
                    f"  Tz: {t_metrics['tz']:.2f}  L2: {t_metrics['l2']:.2f}"
                )
                if "vggt_B_no_flip" in per_subject[subj]:
                    by = per_subject[subj]["vggt_B_no_flip"]["yaw"]
                    bp = per_subject[subj]["vggt_B_no_flip"]["pitch"]
                    br = per_subject[subj]["vggt_B_no_flip"]["roll"]
                    bm = per_subject[subj]["vggt_B_no_flip"]["mean"]
                    line += f"\n  VGGT (B_no_flip) — Yaw: {by:.2f}  Pitch: {bp:.2f}  Roll: {br:.2f}  MAE: {bm:.2f}"
                if sixd_model is not None:
                    sy = s_sixd_y / s_count
                    sp = s_sixd_p / s_count
                    sr = s_sixd_r / s_count
                    sm = (sy + sp + sr) / 3.0
                    per_subject[subj]["sixdrepnet"] = {"yaw": sy, "pitch": sp, "roll": sr, "mean": sm}
                    line += f"\n  6DRepNet          — Yaw: {sy:.2f}  Pitch: {sp:.2f}  Roll: {sr:.2f}  MAE: {sm:.2f}"
                if s_tokenhpe_count > 0:
                    ty = s_tokenhpe_y / s_tokenhpe_count
                    tp = s_tokenhpe_p / s_tokenhpe_count
                    tr = s_tokenhpe_r / s_tokenhpe_count
                    tm = (ty + tp + tr) / 3.0
                    per_subject[subj]["tokenhpe"] = {"yaw": ty, "pitch": tp, "roll": tr, "mean": tm, "count": s_tokenhpe_count}
                    line += f"\n  TokenHPE          — Yaw: {ty:.2f}  Pitch: {tp:.2f}  Roll: {tr:.2f}  MAE: {tm:.2f}  (n={s_tokenhpe_count})"
                if s_whenet_count > 0:
                    wy = s_whenet_y / s_whenet_count
                    wp = s_whenet_p / s_whenet_count
                    wr = s_whenet_r / s_whenet_count
                    wm = (wy + wp + wr) / 3.0
                    per_subject[subj]["whenet"] = {"yaw": wy, "pitch": wp, "roll": wr, "mean": wm, "count": s_whenet_count}
                    line += f"\n  WHENet            — Yaw: {wy:.2f}  Pitch: {wp:.2f}  Roll: {wr:.2f}  MAE: {wm:.2f}  (n={s_whenet_count})"
                if s_trg_count > 0:
                    gy = s_trg_y / s_trg_count
                    gp = s_trg_p / s_trg_count
                    gr = s_trg_r / s_trg_count
                    gm = (gy + gp + gr) / 3.0
                    g_t_metrics = vggt_translation.translation_metrics_from_acc(s_trg_t, s_trg_count)
                    per_subject[subj]["trg"] = {
                        "yaw": gy, "pitch": gp, "roll": gr, "mean": gm,
                        "tx": g_t_metrics["tx"], "ty": g_t_metrics["ty"], "tz": g_t_metrics["tz"], "l2": g_t_metrics["l2"],
                        "count": s_trg_count,
                    }
                    line += (
                        f"\n  TRG               — Yaw: {gy:.2f}  Pitch: {gp:.2f}  Roll: {gr:.2f}  MAE: {gm:.2f}  (n={s_trg_count})"
                        f"\n  TRG Translation   — Tx: {g_t_metrics['tx']:.2f}  Ty: {g_t_metrics['ty']:.2f}"
                        f"  Tz: {g_t_metrics['tz']:.2f}  L2: {g_t_metrics['l2']:.2f}"
                    )
                logging.info(line)

            total_count += s_count
            gc.collect()
            torch.cuda.empty_cache()
            continue

        if use_anchor_pair:
            if args.anchor_selection == "fixed_first":
                anchor_data = get_anchor_data(0)
                if anchor_data is None:
                    logging.warning(f"Subject {subj}: failed to initialize fixed anchor frame, skipping.")
                    continue
                anchor_tensor, R_anchor_t, t_anchor_t, fixed_anchor_frame_num, fixed_anchor_img_rgb = anchor_data
            elif args.anchor_selection == "autoregressive":
                anchor_data = get_anchor_data(0)
                if anchor_data is None:
                    logging.warning(f"Subject {subj}: failed to initialize autoregressive anchor frame, skipping.")
                    continue
                (
                    ar_anchor_tensor,
                    ar_anchor_R_gt,
                    ar_anchor_t_gt,
                    ar_anchor_frame_num,
                    ar_anchor_img_rgb,
                ) = anchor_data
                # Initialize chain with GT pose on frame 0; later this becomes previous prediction.
                ar_anchor_R_pred = ar_anchor_R_gt.clone()
                ar_anchor_t_pred = ar_anchor_t_gt.clone()
            elif args.anchor_selection in {"random_far", "uniform_gap", "close_pose"}:
                min_frame_gap = (
                    args.min_anchor_frame_gap
                    if args.anchor_selection in {"random_far", "uniform_gap"}
                    else args.close_anchor_min_frame_gap
                )
                random_anchor_candidates = _build_random_anchor_candidates(
                    frames,
                    min_frame_gap,
                )
                if len(random_anchor_candidates) == 0:
                    logging.warning(
                        f"Subject {subj}: no valid target/anchor pairs satisfy "
                        f"|Δframe| >= {min_frame_gap}, skipping."
                    )
                    continue
                subject_seed = None if args.anchor_seed < 0 else int(args.anchor_seed) + int(subj)
                subject_rng = np.random.default_rng(subject_seed)
                if args.anchor_selection == "uniform_gap":
                    if anchor_gap_hist is not None:
                        uniform_edges = anchor_gap_hist["edges"]
                        uniform_global_counts = anchor_gap_hist["count"].copy()
                    else:
                        uniform_edges = _init_anchor_gap_hist(args.anchor_gap_bins_deg)["edges"]
                        uniform_global_counts = np.zeros(uniform_edges.size - 1, dtype=np.int64)
                    uniform_subject_counts = np.zeros(uniform_edges.size - 1, dtype=np.int64)
            else:
                raise ValueError(f"Unknown anchor_selection: {args.anchor_selection}")

        R_cam_t = torch.from_numpy(R_cam).float()
        t_cam_t = torch.from_numpy(t_cam).float()
        mtcnn_prev_box = None

        if use_anchor_pair:
            if args.anchor_selection == "fixed_first":
                target_entries = [(idx, *frames[idx]) for idx in range(1, len(frames))]
            elif args.anchor_selection == "autoregressive":
                target_entries = [(idx, *frames[idx]) for idx in range(1, len(frames))]
            else:
                target_entries = [
                    (idx, *frames[idx])
                    for idx in range(len(frames))
                    if idx in random_anchor_candidates
                ]
        else:
            target_entries = [(idx, *frames[idx]) for idx in range(len(frames))]

        n_targets = len(target_entries)
        if n_targets == 0:
            logging.warning(f"Subject {subj}: no valid target frames after anchor selection, skipping.")
            continue
        eval_batch_size = 1 if (use_anchor_pair and args.anchor_selection == "autoregressive") else args.batch_size
        n_batches = (n_targets + eval_batch_size - 1) // eval_batch_size

        if use_anchor_pair:
            if args.anchor_selection == "fixed_first":
                logging.info(
                    f"[{subj}] anchor=frame {fixed_anchor_frame_num}, {n_targets} targets, {n_batches} batches"
                )
            elif args.anchor_selection == "autoregressive":
                logging.info(
                    f"[{subj}] autoregressive anchors (prev prediction), "
                    f"{n_targets} targets, {n_batches} batches (batch_size forced to 1)"
                )
            elif args.anchor_selection == "uniform_gap":
                logging.info(
                    f"[{subj}] uniform-gap anchors |Δframe|>={args.min_anchor_frame_gap}, "
                    f"{n_targets} targets, {n_batches} batches"
                )
            elif args.anchor_selection == "close_pose":
                logging.info(
                    f"[{subj}] close-pose anchors |Δframe|>={args.close_anchor_min_frame_gap}, "
                    f"ΔR<={args.max_anchor_pose_gap_deg:.1f}°, {n_targets} targets, {n_batches} batches"
                )
            else:
                logging.info(
                    f"[{subj}] random anchors |Δframe|>={args.min_anchor_frame_gap}, "
                    f"{n_targets} targets, {n_batches} batches"
                )
        else:
            logging.info(f"[{subj}] single-view evaluation, {n_targets} frames, {n_batches} batches")

        s_variant_errors = base._empty_variant_acc(variant_names)
        s_sixd_y = s_sixd_p = s_sixd_r = 0.0
        s_tokenhpe_y = s_tokenhpe_p = s_tokenhpe_r = 0.0
        s_whenet_y = s_whenet_p = s_whenet_r = 0.0
        s_trg_y = s_trg_p = s_trg_r = 0.0
        s_tokenhpe_count = 0
        s_whenet_count = 0
        s_trg_count = 0
        s_trg_t = vggt_translation._empty_translation_acc()
        s_vggt_t = vggt_translation._empty_translation_acc()
        s_anchor_gap_hist = None
        if anchor_gap_hist is not None:
            s_anchor_gap_hist = _init_anchor_gap_hist(args.anchor_gap_bins_deg)
        s_count = 0
        s_intersection_count = 0
        s_intersection_sums = (
            {m: {"yaw": 0.0, "pitch": 0.0, "roll": 0.0} for m in intersection_methods}
            if intersection_enabled
            else None
        )

        for b_idx, b_start in enumerate(range(0, n_targets, eval_batch_size)):
            batch = target_entries[b_start : b_start + eval_batch_size]

            if b_idx % 20 == 0:
                logging.info(f"  batch {b_idx + 1}/{n_batches}")

            vggt_tensors, sixd_tensors, tokenhpe_tensors, whenet_inputs, trg_tensors = [], [], [], [], []
            trg_bbox_infos, trg_intrinsics = [], []
            batch_anchor_tensors, batch_anchor_R, batch_anchor_t = [], [], []
            batch_anchor_gt_R, batch_anchor_gt_t = [], []
            batch_anchor_frame_nums, batch_anchor_imgs_rgb = [], []
            frame_nums, R_targets, t_targets = [], [], []
            imgs_rgb = []

            for target_idx, frame_num, rgb_path, pose_path in batch:
                anchor_tensor_i = None
                R_anchor_i = None
                t_anchor_i = None
                R_anchor_gt_i = None
                t_anchor_gt_i = None
                anchor_frame_num_i = None
                anchor_img_rgb_i = None
                uniform_bin_i = None
                if use_anchor_pair:
                    if args.anchor_selection == "fixed_first":
                        anchor_tensor_i = anchor_tensor
                        R_anchor_i = R_anchor_t
                        t_anchor_i = t_anchor_t
                        R_anchor_gt_i = R_anchor_t
                        t_anchor_gt_i = t_anchor_t
                        anchor_frame_num_i = fixed_anchor_frame_num
                        anchor_img_rgb_i = fixed_anchor_img_rgb
                    elif args.anchor_selection == "autoregressive":
                        if ar_anchor_tensor is None or ar_anchor_R_pred is None or ar_anchor_t_pred is None:
                            logging.warning(
                                f"Subject {subj} frame {frame_num}: autoregressive anchor state missing, skipping frame."
                            )
                            continue
                        anchor_tensor_i = ar_anchor_tensor
                        R_anchor_i = ar_anchor_R_pred
                        t_anchor_i = ar_anchor_t_pred
                        R_anchor_gt_i = ar_anchor_R_gt
                        t_anchor_gt_i = ar_anchor_t_gt
                        anchor_frame_num_i = ar_anchor_frame_num
                        anchor_img_rgb_i = ar_anchor_img_rgb
                    elif args.anchor_selection == "random_far":
                        candidate_anchor_idxs = list(random_anchor_candidates[target_idx])
                        anchor_data = None
                        while candidate_anchor_idxs:
                            anchor_idx = int(subject_rng.choice(candidate_anchor_idxs))
                            anchor_data = get_anchor_data(anchor_idx)
                            if anchor_data is not None:
                                break
                            candidate_anchor_idxs.remove(anchor_idx)
                        if anchor_data is None:
                            logging.warning(
                                f"Subject {subj} frame {frame_num}: no readable random anchor found, skipping frame."
                            )
                            continue
                        anchor_tensor_i, R_anchor_i, t_anchor_i, anchor_frame_num_i, anchor_img_rgb_i = anchor_data
                        R_anchor_gt_i = R_anchor_i
                        t_anchor_gt_i = t_anchor_i
                    elif args.anchor_selection == "uniform_gap":
                        target_R_i, _ = get_frame_pose(target_idx)
                        best_choice = None
                        for anchor_idx in random_anchor_candidates[target_idx]:
                            anchor_data = get_anchor_data(anchor_idx)
                            if anchor_data is None:
                                continue
                            _, R_a_i, _, _, _ = anchor_data
                            gap_i = _rotation_gap_deg_batch(
                                R_a_i.unsqueeze(0),
                                target_R_i.unsqueeze(0),
                            )[0].item()
                            bi = _bin_index(gap_i, uniform_edges)
                            score = (
                                int(uniform_subject_counts[bi]),
                                int(uniform_global_counts[bi]),
                                float(subject_rng.random()),
                            )
                            if best_choice is None or score < best_choice[0]:
                                best_choice = (score, anchor_idx, anchor_data, bi)

                        if best_choice is None:
                            logging.warning(
                                f"Subject {subj} frame {frame_num}: no readable uniform-gap anchor found, skipping frame."
                            )
                            continue

                        _, _, anchor_data, best_bin = best_choice
                        anchor_tensor_i, R_anchor_i, t_anchor_i, anchor_frame_num_i, anchor_img_rgb_i = anchor_data
                        R_anchor_gt_i = R_anchor_i
                        t_anchor_gt_i = t_anchor_i
                        uniform_bin_i = best_bin
                    elif args.anchor_selection == "close_pose":
                        target_R_i, _ = get_frame_pose(target_idx)
                        best_choice = None
                        for anchor_idx in random_anchor_candidates[target_idx]:
                            anchor_data = get_anchor_data(anchor_idx)
                            if anchor_data is None:
                                continue
                            _, R_a_i, _, _, _ = anchor_data
                            gap_i = _rotation_gap_deg_batch(
                                R_a_i.unsqueeze(0),
                                target_R_i.unsqueeze(0),
                            )[0].item()
                            if gap_i > args.max_anchor_pose_gap_deg:
                                continue
                            score = (float(gap_i), float(subject_rng.random()))
                            if best_choice is None or score < best_choice[0]:
                                best_choice = (score, anchor_idx, anchor_data, gap_i)

                        if best_choice is None:
                            logging.debug(
                                f"Subject {subj} frame {frame_num}: no anchor within "
                                f"{args.max_anchor_pose_gap_deg:.2f} deg, skipping frame."
                            )
                            continue

                        _, _, anchor_data, _ = best_choice
                        anchor_tensor_i, R_anchor_i, t_anchor_i, anchor_frame_num_i, anchor_img_rgb_i = anchor_data
                        R_anchor_gt_i = R_anchor_i
                        t_anchor_gt_i = t_anchor_i
                    else:
                        raise ValueError(f"Unknown anchor_selection: {args.anchor_selection}")

                R_tgt, t_tgt = get_frame_pose(target_idx)
                img_rgb = base.load_img_rgb(rgb_path)
                img_bgr = cv2.imread(rgb_path)
                if img_bgr is None:
                    logging.warning(f"Failed to read target frame {rgb_path}, skipping frame.")
                    continue

                # ── Shared detection: get the detector box and rectangular crop ──
                rect_crop_bgr, mtcnn_prev_box = base.crop_face_mtcnn(
                    img_bgr,
                    mtcnn_detector,
                    mtcnn_prev_box,
                    crop_size=args.crop_size,
                    ad=args.mtcnn_ad,
                )

                # ── VGGT: square crop from the same box (no aspect-ratio distortion) ──
                sq_crop_bgr = square_crop.square_crop_from_box(
                    img_bgr, mtcnn_prev_box, crop_size=args.crop_size,
                )
                sq_crop_rgb = cv2.cvtColor(sq_crop_bgr, cv2.COLOR_BGR2RGB)
                vggt_tensors.append(base.vggt_transform(sq_crop_rgb))

                # ── 6DRepNet: rectangular crop as-is (matches training distortion) ──
                if sixd_model is not None:
                    sixd_tensors.append(base.SIXD_TRANSFORM(Image.fromarray(rect_crop_bgr)))
                # ── TokenHPE: same rect crop, then official resize/crop pipeline ──
                if tokenhpe_model is not None:
                    tokenhpe_tensors.append(TOKENHPE_TRANSFORM(Image.fromarray(rect_crop_bgr)))
                # ── WHENet: rectangular crop as-is, resize 224x224 (official demo behavior) ──
                if whenet_model is not None:
                    rect_crop_rgb = cv2.cvtColor(rect_crop_bgr, cv2.COLOR_BGR2RGB)
                    whenet_inputs.append(cv2.resize(rect_crop_rgb, (224, 224), interpolation=cv2.INTER_LINEAR))
                # ── TRG: square affine crop + matched bbox/intrinsics (closer to native TRG BIWI path) ──
                if trg_model is not None:
                    img_h, img_w = img_bgr.shape[:2]
                    if args.trg_shared_crop_source == "trg_fan":
                        frame_key = f"{subj}/frame_{int(frame_num):05d}_rgb.png"
                        pred_kpt_xy = None if trg_fan_pred_kpt_map is None else trg_fan_pred_kpt_map.get(frame_key)
                        if pred_kpt_xy is not None:
                            left, top, right, bottom = _build_trg_bbox_from_fan_kpt(
                                pred_kpt_xy,
                                img_w,
                                img_h,
                            )
                            trg_fan_bbox_hit += 1
                        else:
                            left, top, right, bottom = _build_trg_square_bbox_from_mtcnn_box(
                                mtcnn_prev_box, img_w, img_h
                            )
                            trg_fan_bbox_fallback += 1
                    else:
                        left, top, right, bottom = _build_trg_square_bbox_from_mtcnn_box(
                            mtcnn_prev_box, img_w, img_h
                        )
                    trg_crop_rgb = _trg_affine_crop(
                        img_bgr, (left, top, right, bottom), img_size=args.trg_img_size
                    )
                    trg_tensors.append(TRG_TRANSFORM(Image.fromarray(trg_crop_rgb)))
                    bbox_info_i = np.array(
                        [float(left), float(top), float(right), float(bottom), float(K_rgb[0, 0]), float(img_h), float(img_w)],
                        dtype=np.float32,
                    )
                    K_crop_i = _build_trg_intrinsic_crop(K_rgb, bbox_info_i, args.trg_img_size)
                    trg_bbox_infos.append(torch.from_numpy(bbox_info_i).float())
                    trg_intrinsics.append(torch.from_numpy(K_crop_i).float())

                frame_nums.append(frame_num)
                R_targets.append(R_tgt)
                t_targets.append(t_tgt)
                imgs_rgb.append(img_rgb)
                if use_anchor_pair:
                    batch_anchor_tensors.append(anchor_tensor_i)
                    batch_anchor_R.append(R_anchor_i)
                    batch_anchor_t.append(t_anchor_i)
                    batch_anchor_gt_R.append(R_anchor_gt_i)
                    batch_anchor_gt_t.append(t_anchor_gt_i)
                    batch_anchor_frame_nums.append(anchor_frame_num_i)
                    batch_anchor_imgs_rgb.append(anchor_img_rgb_i)
                    if args.anchor_selection == "uniform_gap" and uniform_bin_i is not None:
                        uniform_subject_counts[uniform_bin_i] += 1
                        uniform_global_counts[uniform_bin_i] += 1

            if len(vggt_tensors) == 0:
                continue

            bs = len(vggt_tensors)
            R_gt_batch = torch.stack(R_targets)
            t_gt_batch = torch.stack(t_targets)
            y_gt, p_gt, r_gt = base.biwi_euler_deg_batch(R_gt_batch)

            tgt_stack = torch.stack(vggt_tensors)
            if use_anchor_pair:
                if args.anchor_selection == "fixed_first":
                    anchor_rep = anchor_tensor.unsqueeze(0).expand(bs, -1, -1, -1)
                    R_anchor_batch = R_anchor_t
                    t_anchor_batch = t_anchor_t
                    R_anchor_gt_batch = R_anchor_t
                else:
                    anchor_rep = torch.stack(batch_anchor_tensors)
                    R_anchor_batch = torch.stack(batch_anchor_R)
                    t_anchor_batch = torch.stack(batch_anchor_t)
                    R_anchor_gt_batch = torch.stack(batch_anchor_gt_R)
                vggt_images = torch.stack([anchor_rep, tgt_stack], dim=1).to(device)
            else:
                vggt_images = tgt_stack.unsqueeze(1).to(device)
                R_anchor_batch = None
                t_anchor_batch = None
                R_anchor_gt_batch = None

            with torch.no_grad(), torch.cuda.amp.autocast(dtype=amp_dtype):
                vggt_out = vggt_model(images=vggt_images)

            decode_out = vggt_translation.decode_vggt_predictions(
                vggt_out["pose_enc"],
                R_anchor_batch,
                t_anchor_batch,
                R_cam_t,
                t_cam_t,
                pose_mode=args.vggt_pose_mode,
                translation_scale=args.vggt_translation_scale,
                return_debug=args.visualize,
            )
            if args.visualize:
                variants, t_vggt_primary, decode_debug = decode_out
            else:
                variants, t_vggt_primary = decode_out
                decode_debug = None
            if args.vggt_prediction_head_yz_flip:
                variants = _apply_head_yz_flip_to_vggt_variants(variants)

            for vname, R_v in variants.items():
                yv, pv, rv = base.biwi_euler_deg_batch(R_v)
                err_y = base.angular_error(y_gt, yv).sum().item()
                err_p = base.angular_error(p_gt, pv).sum().item()
                err_r = base.angular_error(r_gt, rv).sum().item()
                global_variant_errors[vname]["y"] += err_y
                global_variant_errors[vname]["p"] += err_p
                global_variant_errors[vname]["r"] += err_r
                s_variant_errors[vname]["y"] += err_y
                s_variant_errors[vname]["p"] += err_p
                s_variant_errors[vname]["r"] += err_r

            R_vggt_primary = variants[primary_variant]
            y_vp, p_vp, r_vp = base.biwi_euler_deg_batch(R_vggt_primary)
            err_v_y = base.angular_error(y_gt, y_vp)
            err_v_p = base.angular_error(p_gt, p_vp)
            err_v_r = base.angular_error(r_gt, r_vp)
            err_v_mae = (err_v_y + err_v_p + err_v_r) / 3.0
            err_v_t_abs = (t_vggt_primary - t_gt_batch).abs()
            err_v_t_x = err_v_t_abs[:, 0]
            err_v_t_y = err_v_t_abs[:, 1]
            err_v_t_z = err_v_t_abs[:, 2]
            err_v_t_l2 = (t_vggt_primary - t_gt_batch).norm(dim=-1)
            vggt_tot_y += err_v_y.sum().item()
            vggt_tot_p += err_v_p.sum().item()
            vggt_tot_r += err_v_r.sum().item()
            vggt_translation.accumulate_translation_errors(vggt_tot_t, t_vggt_primary, t_gt_batch)
            vggt_translation.accumulate_translation_errors(s_vggt_t, t_vggt_primary, t_gt_batch)

            y_sp = p_sp = r_sp = None
            R_sixd_biwi = None
            err_sixd_y = err_sixd_p = err_sixd_r = None
            err_sixd_mae = None
            if sixd_model is not None:
                sixd_imgs = torch.stack(sixd_tensors).to(device)
                with torch.no_grad():
                    R_sixd_batch = sixd_model(sixd_imgs).cpu()
                y_sp, p_sp, r_sp, R_sixd_biwi = base.decode_sixdrepnet_pose(R_sixd_batch)

                err_sixd_y = base.angular_error(y_gt, y_sp)
                err_sixd_p = base.angular_error(p_gt, p_sp)
                err_sixd_r = base.angular_error(r_gt, r_sp)
                err_sixd_mae = (err_sixd_y + err_sixd_p + err_sixd_r) / 3.0
                s_sixd_y += err_sixd_y.sum().item()
                s_sixd_p += err_sixd_p.sum().item()
                s_sixd_r += err_sixd_r.sum().item()
                sixd_tot_y += err_sixd_y.sum().item()
                sixd_tot_p += err_sixd_p.sum().item()
                sixd_tot_r += err_sixd_r.sum().item()

            tokenhpe_pred_eulers = [None] * bs
            tokenhpe_pred_R = [None] * bs
            y_th = p_th = r_th = None
            R_tokenhpe_biwi = None
            err_tokenhpe_y = err_tokenhpe_p = err_tokenhpe_r = None
            err_tokenhpe_mae = None
            if tokenhpe_model is not None:
                tokenhpe_imgs = torch.stack(tokenhpe_tensors).to(device)
                with torch.no_grad():
                    R_tokenhpe_batch, _ = tokenhpe_model(tokenhpe_imgs)
                    R_tokenhpe_batch = R_tokenhpe_batch.cpu()
                y_th, p_th, r_th, R_tokenhpe_biwi = base.decode_sixdrepnet_pose(R_tokenhpe_batch)

                err_tokenhpe_y = base.angular_error(y_gt, y_th)
                err_tokenhpe_p = base.angular_error(p_gt, p_th)
                err_tokenhpe_r = base.angular_error(r_gt, r_th)
                err_tokenhpe_mae = (err_tokenhpe_y + err_tokenhpe_p + err_tokenhpe_r) / 3.0
                s_tokenhpe_y += err_tokenhpe_y.sum().item()
                s_tokenhpe_p += err_tokenhpe_p.sum().item()
                s_tokenhpe_r += err_tokenhpe_r.sum().item()
                tokenhpe_tot_y += err_tokenhpe_y.sum().item()
                tokenhpe_tot_p += err_tokenhpe_p.sum().item()
                tokenhpe_tot_r += err_tokenhpe_r.sum().item()
                s_tokenhpe_count += bs
                tokenhpe_eval_count += bs
                for fi in range(bs):
                    tokenhpe_pred_eulers[fi] = (y_th[fi].item(), p_th[fi].item(), r_th[fi].item())
                    tokenhpe_pred_R[fi] = R_tokenhpe_biwi[fi]
            elif tokenhpe_native_by_frame:
                err_tokenhpe_y = torch.full((bs,), float("nan"), dtype=torch.float32)
                err_tokenhpe_p = torch.full((bs,), float("nan"), dtype=torch.float32)
                err_tokenhpe_r = torch.full((bs,), float("nan"), dtype=torch.float32)
                for fi, frame_num_i in enumerate(frame_nums):
                    rec = tokenhpe_native_by_frame.get((subj, int(frame_num_i)))
                    if rec is None:
                        continue
                    yy, pp, rr = rec["euler"]
                    tokenhpe_pred_eulers[fi] = (yy, pp, rr)
                    tokenhpe_pred_R[fi] = rec["R_biwi"]
                    yy_t = torch.tensor([yy], dtype=torch.float32)
                    pp_t = torch.tensor([pp], dtype=torch.float32)
                    rr_t = torch.tensor([rr], dtype=torch.float32)
                    err_t_y_i = base.angular_error(y_gt[fi : fi + 1], yy_t)
                    err_t_p_i = base.angular_error(p_gt[fi : fi + 1], pp_t)
                    err_t_r_i = base.angular_error(r_gt[fi : fi + 1], rr_t)
                    err_tokenhpe_y[fi] = err_t_y_i[0]
                    err_tokenhpe_p[fi] = err_t_p_i[0]
                    err_tokenhpe_r[fi] = err_t_r_i[0]
                    s_tokenhpe_y += err_t_y_i.item()
                    s_tokenhpe_p += err_t_p_i.item()
                    s_tokenhpe_r += err_t_r_i.item()
                    tokenhpe_tot_y += err_t_y_i.item()
                    tokenhpe_tot_p += err_t_p_i.item()
                    tokenhpe_tot_r += err_t_r_i.item()
                    s_tokenhpe_count += 1
                    tokenhpe_eval_count += 1
                err_tokenhpe_mae = (err_tokenhpe_y + err_tokenhpe_p + err_tokenhpe_r) / 3.0

            whenet_pred_eulers = [None] * bs
            whenet_pred_R = [None] * bs
            err_whenet_y = err_whenet_p = err_whenet_r = None
            err_whenet_mae = None
            if whenet_model is not None:
                whenet_batch = np.stack(whenet_inputs, axis=0).astype(np.float32)
                y_wh_np, p_wh_np, r_wh_np = whenet_model.get_angle(whenet_batch)
                y_wh = torch.from_numpy(np.asarray(y_wh_np, dtype=np.float32))
                p_wh = torch.from_numpy(np.asarray(p_wh_np, dtype=np.float32))
                r_wh = torch.from_numpy(np.asarray(r_wh_np, dtype=np.float32))

                err_whenet_y = base.angular_error(y_gt, y_wh)
                err_whenet_p = base.angular_error(p_gt, p_wh)
                err_whenet_r = base.angular_error(r_gt, r_wh)
                err_whenet_mae = (err_whenet_y + err_whenet_p + err_whenet_r) / 3.0
                s_whenet_y += err_whenet_y.sum().item()
                s_whenet_p += err_whenet_p.sum().item()
                s_whenet_r += err_whenet_r.sum().item()
                whenet_tot_y += err_whenet_y.sum().item()
                whenet_tot_p += err_whenet_p.sum().item()
                whenet_tot_r += err_whenet_r.sum().item()
                s_whenet_count += bs
                whenet_eval_count += bs

                R_whenet_biwi = np.stack(
                    [
                        base.euler_to_R_biwi(
                            float(y_wh_np[fi]),
                            float(p_wh_np[fi]),
                            float(r_wh_np[fi]),
                        )
                        for fi in range(bs)
                    ],
                    axis=0,
                )
                for fi in range(bs):
                    whenet_pred_eulers[fi] = (
                        float(y_wh_np[fi]),
                        float(p_wh_np[fi]),
                        float(r_wh_np[fi]),
                    )
                    whenet_pred_R[fi] = R_whenet_biwi[fi]

            trg_pred_eulers = [None] * bs
            trg_pred_R = [None] * bs
            trg_pred_t = [None] * bs
            y_trg = p_trg = r_trg = None
            t_trg_biwi = None
            R_trg_biwi = None
            err_trg_y = err_trg_p = err_trg_r = None
            err_trg_mae = None
            err_trg_t_x = err_trg_t_y = err_trg_t_z = err_trg_t_l2 = None
            if trg_model is not None:
                trg_imgs = torch.stack(trg_tensors).to(device)
                trg_bbox = torch.stack(trg_bbox_infos).to(device)
                trg_intr = torch.stack(trg_intrinsics).to(device)
                with torch.no_grad():
                    preds_dict, _ = trg_model(trg_imgs, trg_intr, trg_bbox)
                    pred_R_t = preds_dict["output"][-1]["pred_R_t"].detach().cpu()

                # TRG predicts in RGB-camera coordinates: R_rgb = R_ext @ R_biwi_pose.
                # Convert back to BIWI pose.txt space for apples-to-apples comparison.
                pred_M = pred_R_t.transpose(1, 2)
                R_pred_rgb = pred_M[:, :3, :3]
                t_pred_rgb_mm = pred_M[:, :3, 3] * 1000.0
                R_ext_t = torch.from_numpy(R_cam).float()
                t_ext_t = torch.from_numpy(t_cam).float()
                R_trg_biwi = torch.matmul(R_ext_t.T.unsqueeze(0), R_pred_rgb)
                t_trg_biwi = torch.matmul(
                    R_ext_t.T.unsqueeze(0),
                    (t_pred_rgb_mm - t_ext_t.unsqueeze(0)).unsqueeze(-1),
                ).squeeze(-1)

                y_trg, p_trg, r_trg = base.biwi_euler_deg_batch(R_trg_biwi)
                err_trg_y = base.angular_error(y_gt, y_trg)
                err_trg_p = base.angular_error(p_gt, p_trg)
                err_trg_r = base.angular_error(r_gt, r_trg)
                err_trg_mae = (err_trg_y + err_trg_p + err_trg_r) / 3.0
                err_trg_t_abs = (t_trg_biwi - t_gt_batch).abs()
                err_trg_t_x = err_trg_t_abs[:, 0]
                err_trg_t_y = err_trg_t_abs[:, 1]
                err_trg_t_z = err_trg_t_abs[:, 2]
                err_trg_t_l2 = (t_trg_biwi - t_gt_batch).norm(dim=-1)
                s_trg_y += err_trg_y.sum().item()
                s_trg_p += err_trg_p.sum().item()
                s_trg_r += err_trg_r.sum().item()
                trg_tot_y += err_trg_y.sum().item()
                trg_tot_p += err_trg_p.sum().item()
                trg_tot_r += err_trg_r.sum().item()
                vggt_translation.accumulate_translation_errors(s_trg_t, t_trg_biwi, t_gt_batch)
                vggt_translation.accumulate_translation_errors(trg_tot_t, t_trg_biwi, t_gt_batch)
                s_trg_count += bs
                trg_eval_count += bs
                for fi in range(bs):
                    trg_pred_eulers[fi] = (y_trg[fi].item(), p_trg[fi].item(), r_trg[fi].item())
                    trg_pred_R[fi] = R_trg_biwi[fi].numpy()
                    trg_pred_t[fi] = t_trg_biwi[fi].numpy()
            elif trg_native_by_frame:
                err_trg_y = torch.full((bs,), float("nan"), dtype=torch.float32)
                err_trg_p = torch.full((bs,), float("nan"), dtype=torch.float32)
                err_trg_r = torch.full((bs,), float("nan"), dtype=torch.float32)
                err_trg_t_x = torch.full((bs,), float("nan"), dtype=torch.float32)
                err_trg_t_y = torch.full((bs,), float("nan"), dtype=torch.float32)
                err_trg_t_z = torch.full((bs,), float("nan"), dtype=torch.float32)
                err_trg_t_l2 = torch.full((bs,), float("nan"), dtype=torch.float32)
                for fi, frame_num_i in enumerate(frame_nums):
                    rec = trg_native_by_frame.get((subj, int(frame_num_i)))
                    if rec is None:
                        continue
                    yy, pp, rr = rec["euler"]
                    tt = rec["t_biwi"]
                    trg_pred_eulers[fi] = (yy, pp, rr)
                    trg_pred_R[fi] = rec["R_biwi"]
                    trg_pred_t[fi] = tt

                    yy_t = torch.tensor([yy], dtype=torch.float32)
                    pp_t = torch.tensor([pp], dtype=torch.float32)
                    rr_t = torch.tensor([rr], dtype=torch.float32)
                    err_g_y_i = base.angular_error(y_gt[fi : fi + 1], yy_t)
                    err_g_p_i = base.angular_error(p_gt[fi : fi + 1], pp_t)
                    err_g_r_i = base.angular_error(r_gt[fi : fi + 1], rr_t)
                    err_trg_y[fi] = err_g_y_i[0]
                    err_trg_p[fi] = err_g_p_i[0]
                    err_trg_r[fi] = err_g_r_i[0]

                    pred_t_i = torch.from_numpy(np.asarray(tt, dtype=np.float32)).view(1, 3)
                    gt_t_i = t_gt_batch[fi : fi + 1]
                    vggt_translation.accumulate_translation_errors(s_trg_t, pred_t_i, gt_t_i)
                    vggt_translation.accumulate_translation_errors(trg_tot_t, pred_t_i, gt_t_i)
                    t_delta_i = (pred_t_i - gt_t_i).abs()
                    err_trg_t_x[fi] = t_delta_i[0, 0]
                    err_trg_t_y[fi] = t_delta_i[0, 1]
                    err_trg_t_z[fi] = t_delta_i[0, 2]
                    err_trg_t_l2[fi] = (pred_t_i - gt_t_i).norm(dim=-1)[0]

                    s_trg_y += err_g_y_i.item()
                    s_trg_p += err_g_p_i.item()
                    s_trg_r += err_g_r_i.item()
                    trg_tot_y += err_g_y_i.item()
                    trg_tot_p += err_g_p_i.item()
                    trg_tot_r += err_g_r_i.item()
                    s_trg_count += 1
                    trg_eval_count += 1
                err_trg_mae = (err_trg_y + err_trg_p + err_trg_r) / 3.0

            inter_mask = None
            if intersection_enabled:
                batch_err = {
                    "vggt": (err_v_y, err_v_p, err_v_r),
                }
                if "sixdrepnet" in intersection_methods:
                    batch_err["sixdrepnet"] = (err_sixd_y, err_sixd_p, err_sixd_r)
                if "tokenhpe" in intersection_methods:
                    batch_err["tokenhpe"] = (err_tokenhpe_y, err_tokenhpe_p, err_tokenhpe_r)
                if "whenet" in intersection_methods:
                    batch_err["whenet"] = (err_whenet_y, err_whenet_p, err_whenet_r)
                if "trg" in intersection_methods:
                    batch_err["trg"] = (err_trg_y, err_trg_p, err_trg_r)

                inter_mask = torch.ones((bs,), dtype=torch.bool)
                for m in intersection_methods:
                    ey, ep, er = batch_err[m]
                    if ey is None or ep is None or er is None:
                        inter_mask &= torch.zeros((bs,), dtype=torch.bool)
                        break
                    inter_mask &= torch.isfinite(ey) & torch.isfinite(ep) & torch.isfinite(er)

                if inter_mask.any():
                    idx = torch.nonzero(inter_mask, as_tuple=False).squeeze(1)
                    inter_n = int(idx.numel())
                    intersection_global_count += inter_n
                    s_intersection_count += inter_n
                    for m in intersection_methods:
                        ey, ep, er = batch_err[m]
                        intersection_global_sums[m]["yaw"] += ey[idx].sum().item()
                        intersection_global_sums[m]["pitch"] += ep[idx].sum().item()
                        intersection_global_sums[m]["roll"] += er[idx].sum().item()
                        s_intersection_sums[m]["yaw"] += ey[idx].sum().item()
                        s_intersection_sums[m]["pitch"] += ep[idx].sum().item()
                        s_intersection_sums[m]["roll"] += er[idx].sum().item()

            if use_anchor_pair:
                if R_anchor_gt_batch.dim() == 2:
                    R_anchor_hist = R_anchor_gt_batch.unsqueeze(0).expand(bs, -1, -1)
                else:
                    R_anchor_hist = R_anchor_gt_batch
                anchor_gap_deg = _rotation_gap_deg_batch(R_anchor_hist, R_gt_batch)
                if anchor_gap_hist is not None:
                    _update_anchor_gap_hist(
                        anchor_gap_hist,
                        anchor_gap_deg,
                        err_v_mae,
                        err_v_t_l2,
                        sixd_sample_mae=err_sixd_mae,
                        tokenhpe_sample_mae=err_tokenhpe_mae,
                        whenet_sample_mae=err_whenet_mae,
                        trg_sample_mae=err_trg_mae,
                    )
                if s_anchor_gap_hist is not None:
                    _update_anchor_gap_hist(
                        s_anchor_gap_hist,
                        anchor_gap_deg,
                        err_v_mae,
                        err_v_t_l2,
                        sixd_sample_mae=err_sixd_mae,
                        tokenhpe_sample_mae=err_tokenhpe_mae,
                        whenet_sample_mae=err_whenet_mae,
                        trg_sample_mae=err_trg_mae,
                    )
                for fi in range(bs):
                    pair_records_all.append(
                        {
                            "subject": str(subj),
                            "target_frame": int(frame_nums[fi]),
                            "anchor_frame": int(batch_anchor_frame_nums[fi]),
                            "anchor_frame_gap_abs": int(abs(int(frame_nums[fi]) - int(batch_anchor_frame_nums[fi]))),
                            "anchor_selection": str(args.anchor_selection),
                            "anchor_gap_deg": float(anchor_gap_deg[fi].item()),
                            "vggt_yaw_err_deg": _scalar_or_nan(err_v_y, fi),
                            "vggt_pitch_err_deg": _scalar_or_nan(err_v_p, fi),
                            "vggt_roll_err_deg": _scalar_or_nan(err_v_r, fi),
                            "vggt_mae_deg": _scalar_or_nan(err_v_mae, fi),
                            "vggt_tx_err_mm": _scalar_or_nan(err_v_t_x, fi),
                            "vggt_ty_err_mm": _scalar_or_nan(err_v_t_y, fi),
                            "vggt_tz_err_mm": _scalar_or_nan(err_v_t_z, fi),
                            "vggt_t_l2_err_mm": _scalar_or_nan(err_v_t_l2, fi),
                            "sixdrepnet_yaw_err_deg": _scalar_or_nan(err_sixd_y, fi),
                            "sixdrepnet_pitch_err_deg": _scalar_or_nan(err_sixd_p, fi),
                            "sixdrepnet_roll_err_deg": _scalar_or_nan(err_sixd_r, fi),
                            "sixdrepnet_mae_deg": _scalar_or_nan(err_sixd_mae, fi),
                            "tokenhpe_yaw_err_deg": _scalar_or_nan(err_tokenhpe_y, fi),
                            "tokenhpe_pitch_err_deg": _scalar_or_nan(err_tokenhpe_p, fi),
                            "tokenhpe_roll_err_deg": _scalar_or_nan(err_tokenhpe_r, fi),
                            "tokenhpe_mae_deg": _scalar_or_nan(err_tokenhpe_mae, fi),
                            "whenet_yaw_err_deg": _scalar_or_nan(err_whenet_y, fi),
                            "whenet_pitch_err_deg": _scalar_or_nan(err_whenet_p, fi),
                            "whenet_roll_err_deg": _scalar_or_nan(err_whenet_r, fi),
                            "whenet_mae_deg": _scalar_or_nan(err_whenet_mae, fi),
                            "trg_yaw_err_deg": _scalar_or_nan(err_trg_y, fi),
                            "trg_pitch_err_deg": _scalar_or_nan(err_trg_p, fi),
                            "trg_roll_err_deg": _scalar_or_nan(err_trg_r, fi),
                            "trg_mae_deg": _scalar_or_nan(err_trg_mae, fi),
                            "trg_tx_err_mm": _scalar_or_nan(err_trg_t_x, fi),
                            "trg_ty_err_mm": _scalar_or_nan(err_trg_t_y, fi),
                            "trg_tz_err_mm": _scalar_or_nan(err_trg_t_z, fi),
                            "trg_t_l2_err_mm": _scalar_or_nan(err_trg_t_l2, fi),
                            "is_intersection": bool(inter_mask[fi].item()) if inter_mask is not None else False,
                        }
                    )
            else:
                for fi in range(bs):
                    pair_records_all.append(
                        {
                            "subject": str(subj),
                            "target_frame": int(frame_nums[fi]),
                            "anchor_frame": "",
                            "anchor_frame_gap_abs": "",
                            "anchor_selection": "single_view",
                            "anchor_gap_deg": "",
                            "vggt_yaw_err_deg": _scalar_or_nan(err_v_y, fi),
                            "vggt_pitch_err_deg": _scalar_or_nan(err_v_p, fi),
                            "vggt_roll_err_deg": _scalar_or_nan(err_v_r, fi),
                            "vggt_mae_deg": _scalar_or_nan(err_v_mae, fi),
                            "vggt_tx_err_mm": _scalar_or_nan(err_v_t_x, fi),
                            "vggt_ty_err_mm": _scalar_or_nan(err_v_t_y, fi),
                            "vggt_tz_err_mm": _scalar_or_nan(err_v_t_z, fi),
                            "vggt_t_l2_err_mm": _scalar_or_nan(err_v_t_l2, fi),
                            "sixdrepnet_yaw_err_deg": _scalar_or_nan(err_sixd_y, fi),
                            "sixdrepnet_pitch_err_deg": _scalar_or_nan(err_sixd_p, fi),
                            "sixdrepnet_roll_err_deg": _scalar_or_nan(err_sixd_r, fi),
                            "sixdrepnet_mae_deg": _scalar_or_nan(err_sixd_mae, fi),
                            "tokenhpe_yaw_err_deg": _scalar_or_nan(err_tokenhpe_y, fi),
                            "tokenhpe_pitch_err_deg": _scalar_or_nan(err_tokenhpe_p, fi),
                            "tokenhpe_roll_err_deg": _scalar_or_nan(err_tokenhpe_r, fi),
                            "tokenhpe_mae_deg": _scalar_or_nan(err_tokenhpe_mae, fi),
                            "whenet_yaw_err_deg": _scalar_or_nan(err_whenet_y, fi),
                            "whenet_pitch_err_deg": _scalar_or_nan(err_whenet_p, fi),
                            "whenet_roll_err_deg": _scalar_or_nan(err_whenet_r, fi),
                            "whenet_mae_deg": _scalar_or_nan(err_whenet_mae, fi),
                            "trg_yaw_err_deg": _scalar_or_nan(err_trg_y, fi),
                            "trg_pitch_err_deg": _scalar_or_nan(err_trg_p, fi),
                            "trg_roll_err_deg": _scalar_or_nan(err_trg_r, fi),
                            "trg_mae_deg": _scalar_or_nan(err_trg_mae, fi),
                            "trg_tx_err_mm": _scalar_or_nan(err_trg_t_x, fi),
                            "trg_ty_err_mm": _scalar_or_nan(err_trg_t_y, fi),
                            "trg_tz_err_mm": _scalar_or_nan(err_trg_t_z, fi),
                            "trg_t_l2_err_mm": _scalar_or_nan(err_trg_t_l2, fi),
                            "is_intersection": bool(inter_mask[fi].item()) if inter_mask is not None else False,
                        }
                    )

            s_count += bs

            if args.visualize:
                for fi in range(bs):
                    if vis_count % args.vis_every_n == 0:
                        frame_num = frame_nums[fi]
                        save_path = os.path.join(args.vis_dir, f"subj{subj}_frame{frame_num:05d}.png")

                        euler_gt_i = (y_gt[fi].item(), p_gt[fi].item(), r_gt[fi].item())
                        vggt_vis = {}
                        for vname, R_v in variants.items():
                            yv, pv, rv = base.biwi_euler_deg_batch(R_v)
                            vggt_vis[vname] = (
                                R_v[fi].numpy(),
                                (yv[fi].item(), pv[fi].item(), rv[fi].item()),
                            )

                        euler_sixd_i = (
                            (y_sp[fi].item(), p_sp[fi].item(), r_sp[fi].item())
                            if y_sp is not None
                            else None
                        )
                        euler_tokenhpe_i = tokenhpe_pred_eulers[fi]
                        euler_whenet_i = whenet_pred_eulers[fi]
                        euler_trg_i = trg_pred_eulers[fi]
                        prefix_panels = []
                        anchor_y = anchor_p = anchor_r = None
                        if use_anchor_pair:
                            anchor_y, anchor_p, anchor_r = _euler_from_rotation(batch_anchor_R[fi])

                        if use_anchor_pair and batch_anchor_imgs_rgb[fi] is not None:
                            anchor_panel = base.render_mesh_on_image(
                                batch_anchor_imgs_rgb[fi].copy(),
                                vertices,
                                faces,
                                batch_anchor_R[fi].numpy(),
                                batch_anchor_t[fi].numpy(),
                                R_cam,
                                t_cam,
                                K_rgb,
                                color="#4a90d9",
                                alpha=0.5,
                            )
                            prefix_panels.append(
                                (
                                    anchor_panel,
                                    f"Anchor {'Pred' if args.anchor_selection == 'autoregressive' else 'GT'} "
                                    f"f{batch_anchor_frame_nums[fi]}\n"
                                    f"Y:{anchor_y:.1f}  P:{anchor_p:.1f}  R:{anchor_r:.1f}",
                                )
                            )

                        if (
                            use_anchor_pair
                            and decode_debug is not None
                            and args.vggt_pose_mode == "relative_h2c"
                            and "R_rel_depth" in decode_debug
                        ):
                            rel_y, rel_p, rel_r = _euler_from_rotation(decode_debug["R_rel_depth"][fi])
                            final_y, final_p, final_r = _euler_from_rotation(decode_debug["R_final_primary"][fi])
                            decomp_panel = _make_text_panel_like(
                                imgs_rgb[fi],
                                [
                                    "VGGT Relative Decomposition",
                                    f"R_rel (depth/OpenCV): Y:{rel_y:.1f} P:{rel_p:.1f} R:{rel_r:.1f}",
                                    f"R_anchor (used):       Y:{anchor_y:.1f} P:{anchor_p:.1f} R:{anchor_r:.1f}",
                                    "R_final = R_rel @ R_anchor",
                                    f"R_final ({primary_variant}): Y:{final_y:.1f} P:{final_p:.1f} R:{final_r:.1f}",
                                ],
                            )
                            prefix_panels.append((decomp_panel, "Relative -> Final (same frame)"))

                        if len(prefix_panels) == 0:
                            prefix_panels = None

                        base.save_vis_frame(
                            img_rgb=imgs_rgb[fi],
                            vertices=vertices,
                            faces=faces,
                            K_rgb=K_rgb,
                            R_cam=R_cam,
                            t_cam=t_cam,
                            R_gt=R_gt_batch[fi].numpy(),
                            t_gt=t_gt_batch[fi].numpy(),
                            euler_gt=euler_gt_i,
                            vggt_variants=vggt_vis,
                            R_sixd=R_sixd_biwi[fi] if R_sixd_biwi is not None else None,
                            euler_sixd=euler_sixd_i,
                            R_tokenhpe=tokenhpe_pred_R[fi],
                            euler_tokenhpe=euler_tokenhpe_i,
                            R_whenet=whenet_pred_R[fi],
                            euler_whenet=euler_whenet_i,
                            R_trg=trg_pred_R[fi],
                            euler_trg=euler_trg_i,
                            save_path=save_path,
                            prefix_panels=prefix_panels,
                        )
                    vis_count += 1

            if use_anchor_pair and args.anchor_selection == "autoregressive":
                # Sequential mode (batch_size forced to 1): next anchor is current prediction.
                for fi in range(bs):
                    ar_anchor_tensor = tgt_stack[fi].detach().clone()
                    ar_anchor_R_pred = R_vggt_primary[fi].detach().clone()
                    ar_anchor_t_pred = t_vggt_primary[fi].detach().clone()
                    ar_anchor_R_gt = R_gt_batch[fi].detach().clone()
                    ar_anchor_t_gt = t_gt_batch[fi].detach().clone()
                    ar_anchor_frame_num = int(frame_nums[fi])
                    ar_anchor_img_rgb = imgs_rgb[fi] if args.visualize else None

        if s_count > 0:
            per_subject[subj] = {}
            for vname in variant_names:
                vy = s_variant_errors[vname]["y"] / s_count
                vp = s_variant_errors[vname]["p"] / s_count
                vr = s_variant_errors[vname]["r"] / s_count
                vm = (vy + vp + vr) / 3
                per_subject[subj][f"vggt_{vname}"] = {
                    "yaw": vy,
                    "pitch": vp,
                    "roll": vr,
                    "mean": vm,
                }

            t_metrics = vggt_translation.translation_metrics_from_acc(s_vggt_t, s_count)
            per_subject[subj][f"vggt_{primary_variant}"].update(t_metrics)

            vy = per_subject[subj][f"vggt_{primary_variant}"]["yaw"]
            vp = per_subject[subj][f"vggt_{primary_variant}"]["pitch"]
            vr = per_subject[subj][f"vggt_{primary_variant}"]["roll"]
            vm = per_subject[subj][f"vggt_{primary_variant}"]["mean"]
            line = (
                f"Subject {subj} ({s_count} frames):\n"
                f"  VGGT ({primary_variant}) — Yaw: {vy:.2f}  Pitch: {vp:.2f}  Roll: {vr:.2f}  MAE: {vm:.2f}\n"
                f"  VGGT Translation    — Tx: {t_metrics['tx']:.2f}  Ty: {t_metrics['ty']:.2f}"
                f"  Tz: {t_metrics['tz']:.2f}  L2: {t_metrics['l2']:.2f}"
            )

            if "vggt_B_no_flip" in per_subject[subj]:
                by = per_subject[subj]["vggt_B_no_flip"]["yaw"]
                bp = per_subject[subj]["vggt_B_no_flip"]["pitch"]
                br = per_subject[subj]["vggt_B_no_flip"]["roll"]
                bm = per_subject[subj]["vggt_B_no_flip"]["mean"]
                line += (
                    f"\n  VGGT (B_no_flip) — Yaw: {by:.2f}  Pitch: {bp:.2f}"
                    f"  Roll: {br:.2f}  MAE: {bm:.2f}"
                )

            if sixd_model is not None:
                sy = s_sixd_y / s_count
                sp = s_sixd_p / s_count
                sr = s_sixd_r / s_count
                sm = (sy + sp + sr) / 3
                per_subject[subj]["sixdrepnet"] = {
                    "yaw": sy,
                    "pitch": sp,
                    "roll": sr,
                    "mean": sm,
                }
                line += f"\n  6DRepNet          — Yaw: {sy:.2f}  Pitch: {sp:.2f}  Roll: {sr:.2f}  MAE: {sm:.2f}"
            if s_tokenhpe_count > 0:
                ty = s_tokenhpe_y / s_tokenhpe_count
                tp = s_tokenhpe_p / s_tokenhpe_count
                tr = s_tokenhpe_r / s_tokenhpe_count
                tm = (ty + tp + tr) / 3
                per_subject[subj]["tokenhpe"] = {
                    "yaw": ty,
                    "pitch": tp,
                    "roll": tr,
                    "mean": tm,
                    "count": s_tokenhpe_count,
                }
                line += (
                    f"\n  TokenHPE          — Yaw: {ty:.2f}  Pitch: {tp:.2f}  Roll: {tr:.2f}  MAE: {tm:.2f}"
                    f"  (n={s_tokenhpe_count})"
                )
            if s_whenet_count > 0:
                wy = s_whenet_y / s_whenet_count
                wp = s_whenet_p / s_whenet_count
                wr = s_whenet_r / s_whenet_count
                wm = (wy + wp + wr) / 3
                per_subject[subj]["whenet"] = {
                    "yaw": wy,
                    "pitch": wp,
                    "roll": wr,
                    "mean": wm,
                    "count": s_whenet_count,
                }
                line += (
                    f"\n  WHENet            — Yaw: {wy:.2f}  Pitch: {wp:.2f}  Roll: {wr:.2f}  MAE: {wm:.2f}"
                    f"  (n={s_whenet_count})"
                )
            if s_trg_count > 0:
                gy = s_trg_y / s_trg_count
                gp = s_trg_p / s_trg_count
                gr = s_trg_r / s_trg_count
                gm = (gy + gp + gr) / 3
                g_t_metrics = vggt_translation.translation_metrics_from_acc(s_trg_t, s_trg_count)
                per_subject[subj]["trg"] = {
                    "yaw": gy,
                    "pitch": gp,
                    "roll": gr,
                    "mean": gm,
                    "tx": g_t_metrics["tx"],
                    "ty": g_t_metrics["ty"],
                    "tz": g_t_metrics["tz"],
                    "l2": g_t_metrics["l2"],
                    "count": s_trg_count,
                }
                line += (
                    f"\n  TRG               — Yaw: {gy:.2f}  Pitch: {gp:.2f}  Roll: {gr:.2f}  MAE: {gm:.2f}"
                    f"  (n={s_trg_count})"
                    f"\n  TRG Translation   — Tx: {g_t_metrics['tx']:.2f}  Ty: {g_t_metrics['ty']:.2f}"
                    f"  Tz: {g_t_metrics['tz']:.2f}  L2: {g_t_metrics['l2']:.2f}"
                )
            if (trg_model is not None) and (subj in trg_native_per_subject):
                ntrg = trg_native_per_subject[subj]
                per_subject[subj]["trg_native"] = ntrg
                line += (
                    f"\n  TRG (native)      — Yaw: {ntrg['yaw']:.2f}  Pitch: {ntrg['pitch']:.2f}"
                    f"  Roll: {ntrg['roll']:.2f}  MAE: {ntrg['mean']:.2f}  (n={int(ntrg.get('count', 0))})"
                    f"\n  TRG Native Trans  — Tx: {ntrg['tx']:.2f}  Ty: {ntrg['ty']:.2f}"
                    f"  Tz: {ntrg['tz']:.2f}  L2: {ntrg['l2']:.2f}"
                )
            if intersection_enabled and s_intersection_count > 0:
                line += f"\n  Intersection (all active methods) — n={s_intersection_count}"
                for m in intersection_methods:
                    my = s_intersection_sums[m]["yaw"] / s_intersection_count
                    mp = s_intersection_sums[m]["pitch"] / s_intersection_count
                    mr = s_intersection_sums[m]["roll"] / s_intersection_count
                    mm = (my + mp + mr) / 3.0
                    per_subject[subj][f"intersection_{m}"] = {
                        "yaw": my,
                        "pitch": mp,
                        "roll": mr,
                        "mean": mm,
                        "count": s_intersection_count,
                    }
                    line += (
                        f"\n    {_method_display_name(m):18s} — Yaw: {my:.2f}  Pitch: {mp:.2f}"
                        f"  Roll: {mr:.2f}  MAE: {mm:.2f}"
                    )
            if subj in sixdof_per_subject:
                m = sixdof_per_subject[subj]
                per_subject[subj]["sixdof_face"] = m
                line += (
                    f"\n  6DoF Face         — Yaw: {m['yaw']:.2f}  Pitch: {m['pitch']:.2f}"
                    f"  Roll: {m['roll']:.2f}  MAE: {m['mean']:.2f}"
                    f"\n  6DoF Face Trans   — Tx: {m['tx']:.2f}  Ty: {m['ty']:.2f}"
                    f"  Tz: {m['tz']:.2f}  L2: {m['l2']:.2f}"
                )
            logging.info(line)
            if s_anchor_gap_hist is not None:
                subj_stem = f"anchor_gap_error_histogram_subj{subj}"
                hist_csv, hist_png = _save_anchor_gap_histogram(
                    s_anchor_gap_hist,
                    args.vis_dir,
                    file_stem=subj_stem,
                    title_prefix=f"Subject {subj} —",
                )
                if hist_csv is not None and hist_png is not None:
                    logging.info(f"Subject {subj} anchor-gap histogram CSV -> {hist_csv}")
                    logging.info(f"Subject {subj} anchor-gap histogram PNG -> {hist_png}")
                if args.anchor_selection == "uniform_gap":
                    logging.info(f"Subject {subj} anchor-gap counts -> {_format_hist_counts(s_anchor_gap_hist)}")

        total_count += s_count
        gc.collect()
        torch.cuda.empty_cache()

    for subj, metrics in sixdof_per_subject.items():
        per_subject.setdefault(subj, {})
        per_subject[subj]["sixdof_face"] = metrics

    if (trg_model is not None) and (args.trg_shared_crop_source == "trg_fan"):
        logging.info(
            "TRG shared FAN-bbox coverage: hit=%d  fallback_to_mtcnn=%d",
            trg_fan_bbox_hit,
            trg_fan_bbox_fallback,
        )

    print("\n" + "=" * 60)
    print("BIWI EVALUATION RESULTS")
    print("=" * 60)
    if total_count > 0:
        vy = vggt_tot_y / total_count
        vp = vggt_tot_p / total_count
        vr = vggt_tot_r / total_count
        vm = (vy + vp + vr) / 3
        t_metrics = vggt_translation.translation_metrics_from_acc(vggt_tot_t, total_count)

        print(f"Frames evaluated: {total_count}")
        print(f"\nVGGT ({primary_variant}) — Yaw: {vy:.4f}  Pitch: {vp:.4f}  Roll: {vr:.4f}  MAE: {vm:.4f}")
        print(
            f"VGGT Translation    — Tx: {t_metrics['tx']:.4f}  Ty: {t_metrics['ty']:.4f}"
            f"  Tz: {t_metrics['tz']:.4f}  L2: {t_metrics['l2']:.4f}"
        )
        if sixd_model is not None:
            sy = sixd_tot_y / total_count
            sp = sixd_tot_p / total_count
            sr = sixd_tot_r / total_count
            sm = (sy + sp + sr) / 3
            print(f"6DRepNet           — Yaw: {sy:.4f}  Pitch: {sp:.4f}  Roll: {sr:.4f}  MAE: {sm:.4f}")
        if tokenhpe_eval_count > 0:
            ty = tokenhpe_tot_y / tokenhpe_eval_count
            tp = tokenhpe_tot_p / tokenhpe_eval_count
            tr = tokenhpe_tot_r / tokenhpe_eval_count
            tm = (ty + tp + tr) / 3
            print(f"TokenHPE           — Yaw: {ty:.4f}  Pitch: {tp:.4f}  Roll: {tr:.4f}  MAE: {tm:.4f}  (n={tokenhpe_eval_count})")
        if whenet_eval_count > 0:
            wy = whenet_tot_y / whenet_eval_count
            wp = whenet_tot_p / whenet_eval_count
            wr = whenet_tot_r / whenet_eval_count
            wm = (wy + wp + wr) / 3
            print(f"WHENet             — Yaw: {wy:.4f}  Pitch: {wp:.4f}  Roll: {wr:.4f}  MAE: {wm:.4f}  (n={whenet_eval_count})")
        if tokenhpe_native_overall is not None:
            print(
                f"TokenHPE (native)  — Yaw: {tokenhpe_native_overall['yaw']:.4f}  "
                f"Pitch: {tokenhpe_native_overall['pitch']:.4f}  "
                f"Roll: {tokenhpe_native_overall['roll']:.4f}  MAE: {tokenhpe_native_overall['mean']:.4f}"
            )
        if trg_eval_count > 0:
            gy = trg_tot_y / trg_eval_count
            gp = trg_tot_p / trg_eval_count
            gr = trg_tot_r / trg_eval_count
            gm = (gy + gp + gr) / 3
            g_t_metrics = vggt_translation.translation_metrics_from_acc(trg_tot_t, trg_eval_count)
            print(f"TRG                — Yaw: {gy:.4f}  Pitch: {gp:.4f}  Roll: {gr:.4f}  MAE: {gm:.4f}  (n={trg_eval_count})")
            print(
                f"TRG Translation     — Tx: {g_t_metrics['tx']:.4f}  Ty: {g_t_metrics['ty']:.4f}"
                f"  Tz: {g_t_metrics['tz']:.4f}  L2: {g_t_metrics['l2']:.4f}"
            )
        if trg_native_overall is not None:
            print(
                f"TRG (native)       — Yaw: {trg_native_overall['yaw']:.4f}  "
                f"Pitch: {trg_native_overall['pitch']:.4f}  "
                f"Roll: {trg_native_overall['roll']:.4f}  MAE: {trg_native_overall['mean']:.4f}"
            )
            print(
                f"TRG Native Trans    — Tx: {trg_native_overall['tx']:.4f}  Ty: {trg_native_overall['ty']:.4f}"
                f"  Tz: {trg_native_overall['tz']:.4f}  L2: {trg_native_overall['l2']:.4f}"
            )
        if sixdof_overall is not None:
            print(
                f"6DoF Face          — Yaw: {sixdof_overall['yaw']:.4f}  Pitch: {sixdof_overall['pitch']:.4f}"
                f"  Roll: {sixdof_overall['roll']:.4f}  MAE: {sixdof_overall['mean']:.4f}"
            )
            print(
                f"6DoF Face Trans    — Tx: {sixdof_overall['tx']:.4f}  Ty: {sixdof_overall['ty']:.4f}"
                f"  Tz: {sixdof_overall['tz']:.4f}  L2: {sixdof_overall['l2']:.4f}"
            )
        if intersection_enabled and intersection_global_count > 0:
            print(f"\nIntersection (all active methods) — frames: {intersection_global_count}")
            for m in intersection_methods:
                my = intersection_global_sums[m]["yaw"] / intersection_global_count
                mp = intersection_global_sums[m]["pitch"] / intersection_global_count
                mr = intersection_global_sums[m]["roll"] / intersection_global_count
                mm = (my + mp + mr) / 3.0
                print(
                    f"{_method_display_name(m):18s} — Yaw: {my:.4f}  Pitch: {mp:.4f}"
                    f"  Roll: {mr:.4f}  MAE: {mm:.4f}"
                )

        print(f"\n{'─' * 60}")
        print("VGGT variant comparison:")
        print(f"{'─' * 60}")
        best_name, best_mae = None, float("inf")
        for vname in variant_names:
            errs = global_variant_errors[vname]
            vy_ = errs["y"] / total_count
            vp_ = errs["p"] / total_count
            vr_ = errs["r"] / total_count
            vm_ = (vy_ + vp_ + vr_) / 3
            if vm_ < best_mae:
                best_mae = vm_
                best_name = vname
            marker = " <- best" if vname == best_name else ""
            print(f"  {vname:15s} — Yaw: {vy_:.4f}  Pitch: {vp_:.4f}  Roll: {vr_:.4f}  MAE: {vm_:.4f}{marker}")
        print(f"\n  Best variant: {best_name}  (MAE: {best_mae:.4f} deg)")
    else:
        print("No VGGT frames evaluated.")
    print("=" * 60)

    csv_path = os.path.join(args.vis_dir, "biwi_eval_results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        header = [
            "subject",
            "model",
            "yaw_mae",
            "pitch_mae",
            "roll_mae",
            "mean_mae",
            "tx_mae_mm",
            "ty_mae_mm",
            "tz_mae_mm",
            "translation_l2_mae_mm",
        ]
        w.writerow(header)
        for subj, res in sorted(per_subject.items(), key=lambda item: item[0]):
            for model_name, vals in res.items():
                w.writerow(
                    [
                        subj,
                        model_name,
                        f"{vals.get('yaw', float('nan')):.4f}" if "yaw" in vals else "",
                        f"{vals.get('pitch', float('nan')):.4f}" if "pitch" in vals else "",
                        f"{vals.get('roll', float('nan')):.4f}" if "roll" in vals else "",
                        f"{vals.get('mean', float('nan')):.4f}" if "mean" in vals else "",
                        f"{vals.get('tx', float('nan')):.4f}" if "tx" in vals else "",
                        f"{vals.get('ty', float('nan')):.4f}" if "ty" in vals else "",
                        f"{vals.get('tz', float('nan')):.4f}" if "tz" in vals else "",
                        f"{vals.get('l2', float('nan')):.4f}" if "l2" in vals else "",
                    ]
                )

        if total_count > 0:
            t_metrics = vggt_translation.translation_metrics_from_acc(vggt_tot_t, total_count)
            w.writerow([])
            w.writerow(["# Overall results"])
            w.writerow(header)
            w.writerow(
                [
                    "overall",
                    f"vggt_{primary_variant}",
                    f"{vggt_tot_y / total_count:.4f}",
                    f"{vggt_tot_p / total_count:.4f}",
                    f"{vggt_tot_r / total_count:.4f}",
                    f"{(vggt_tot_y + vggt_tot_p + vggt_tot_r) / (total_count * 3):.4f}",
                    f"{t_metrics['tx']:.4f}",
                    f"{t_metrics['ty']:.4f}",
                    f"{t_metrics['tz']:.4f}",
                    f"{t_metrics['l2']:.4f}",
                ]
            )
            if sixd_model is not None:
                w.writerow(
                    [
                        "overall",
                        "sixdrepnet",
                        f"{sixd_tot_y / total_count:.4f}",
                        f"{sixd_tot_p / total_count:.4f}",
                        f"{sixd_tot_r / total_count:.4f}",
                        f"{(sixd_tot_y + sixd_tot_p + sixd_tot_r) / (total_count * 3):.4f}",
                        "",
                        "",
                        "",
                        "",
                    ]
                )
            if tokenhpe_eval_count > 0:
                w.writerow(
                    [
                        "overall",
                        "tokenhpe",
                        f"{tokenhpe_tot_y / tokenhpe_eval_count:.4f}",
                        f"{tokenhpe_tot_p / tokenhpe_eval_count:.4f}",
                        f"{tokenhpe_tot_r / tokenhpe_eval_count:.4f}",
                        f"{(tokenhpe_tot_y + tokenhpe_tot_p + tokenhpe_tot_r) / (tokenhpe_eval_count * 3):.4f}",
                        "",
                        "",
                        "",
                        "",
                    ]
                )
            if whenet_eval_count > 0:
                w.writerow(
                    [
                        "overall",
                        "whenet",
                        f"{whenet_tot_y / whenet_eval_count:.4f}",
                        f"{whenet_tot_p / whenet_eval_count:.4f}",
                        f"{whenet_tot_r / whenet_eval_count:.4f}",
                        f"{(whenet_tot_y + whenet_tot_p + whenet_tot_r) / (whenet_eval_count * 3):.4f}",
                        "",
                        "",
                        "",
                        "",
                    ]
                )
            if tokenhpe_native_overall is not None:
                w.writerow(
                    [
                        "overall",
                        "tokenhpe_native",
                        f"{tokenhpe_native_overall['yaw']:.4f}",
                        f"{tokenhpe_native_overall['pitch']:.4f}",
                        f"{tokenhpe_native_overall['roll']:.4f}",
                        f"{tokenhpe_native_overall['mean']:.4f}",
                        "",
                        "",
                        "",
                        "",
                    ]
                )
            if trg_eval_count > 0:
                g_t_metrics = vggt_translation.translation_metrics_from_acc(trg_tot_t, trg_eval_count)
                w.writerow(
                    [
                        "overall",
                        "trg",
                        f"{trg_tot_y / trg_eval_count:.4f}",
                        f"{trg_tot_p / trg_eval_count:.4f}",
                        f"{trg_tot_r / trg_eval_count:.4f}",
                        f"{(trg_tot_y + trg_tot_p + trg_tot_r) / (trg_eval_count * 3):.4f}",
                        f"{g_t_metrics['tx']:.4f}",
                        f"{g_t_metrics['ty']:.4f}",
                        f"{g_t_metrics['tz']:.4f}",
                        f"{g_t_metrics['l2']:.4f}",
                    ]
                )
            if trg_native_overall is not None:
                w.writerow(
                    [
                        "overall",
                        "trg_native",
                        f"{trg_native_overall['yaw']:.4f}",
                        f"{trg_native_overall['pitch']:.4f}",
                        f"{trg_native_overall['roll']:.4f}",
                        f"{trg_native_overall['mean']:.4f}",
                        f"{trg_native_overall['tx']:.4f}",
                        f"{trg_native_overall['ty']:.4f}",
                        f"{trg_native_overall['tz']:.4f}",
                        f"{trg_native_overall['l2']:.4f}",
                    ]
                )
        if sixdof_overall is not None:
            w.writerow(
                [
                    "overall",
                    "sixdof_face",
                    f"{sixdof_overall['yaw']:.4f}",
                    f"{sixdof_overall['pitch']:.4f}",
                    f"{sixdof_overall['roll']:.4f}",
                    f"{sixdof_overall['mean']:.4f}",
                    f"{sixdof_overall['tx']:.4f}",
                    f"{sixdof_overall['ty']:.4f}",
                    f"{sixdof_overall['tz']:.4f}",
                    f"{sixdof_overall['l2']:.4f}",
                ]
            )

        if total_count > 0:
            w.writerow([])
            w.writerow(["# VGGT variants (rotation only)"])
            w.writerow(["variant", "model", "yaw_mae", "pitch_mae", "roll_mae", "mean_mae"])
            best_name, best_mae = None, float("inf")
            for vname in variant_names:
                errs = global_variant_errors[vname]
                vy_ = errs["y"] / total_count
                vp_ = errs["p"] / total_count
                vr_ = errs["r"] / total_count
                vm_ = (vy_ + vp_ + vr_) / 3
                w.writerow([vname, "vggt", f"{vy_:.4f}", f"{vp_:.4f}", f"{vr_:.4f}", f"{vm_:.4f}"])
                if vm_ < best_mae:
                    best_mae = vm_
                    best_name = vname
            w.writerow(["best_variant", best_name, "", "", "", f"{best_mae:.4f}"])

    logging.info(f"Results saved -> {csv_path}")

    if intersection_enabled and intersection_global_count > 0:
        intersection_csv_path = os.path.join(args.vis_dir, "biwi_eval_results_intersection.csv")
        with open(intersection_csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "scope",
                    "subject",
                    "intersection_count",
                    "model",
                    "yaw_mae",
                    "pitch_mae",
                    "roll_mae",
                    "mean_mae",
                ]
            )
            for subj, res in sorted(per_subject.items(), key=lambda item: item[0]):
                for m in intersection_methods:
                    key = f"intersection_{m}"
                    if key not in res:
                        continue
                    vals = res[key]
                    w.writerow(
                        [
                            "subject",
                            subj,
                            int(vals.get("count", 0)),
                            _method_display_name(m),
                            f"{vals['yaw']:.4f}",
                            f"{vals['pitch']:.4f}",
                            f"{vals['roll']:.4f}",
                            f"{vals['mean']:.4f}",
                        ]
                    )

            for m in intersection_methods:
                my = intersection_global_sums[m]["yaw"] / intersection_global_count
                mp = intersection_global_sums[m]["pitch"] / intersection_global_count
                mr = intersection_global_sums[m]["roll"] / intersection_global_count
                mm = (my + mp + mr) / 3.0
                w.writerow(
                    [
                        "overall",
                        "overall",
                        intersection_global_count,
                        _method_display_name(m),
                        f"{my:.4f}",
                        f"{mp:.4f}",
                        f"{mr:.4f}",
                        f"{mm:.4f}",
                    ]
                )
        logging.info(f"Intersection results saved -> {intersection_csv_path}")

    if len(pair_records_all) > 0:
        pair_meta = {
            "anchor_selection": args.anchor_selection,
            "anchor_gap_bins_deg": [float(x) for x in args.anchor_gap_bins_deg],
            "vggt_pose_mode": args.vggt_pose_mode,
            "face_detector": args.face_detector,
            "detector_crop_expansion": float(args.mtcnn_ad),
            "yunet_score_threshold": (
                float(args.yunet_score_threshold) if args.face_detector == "yunet" else None
            ),
            "intersection_enabled": bool(intersection_enabled),
            "intersection_methods": list(intersection_methods),
            "note": "Normal set = all valid anchor-target pairs. Intersection set = rows where all active intersection methods have finite errors.",
        }
        normal_csv, normal_pkl = _save_pair_records(
            pair_records_all,
            args.vis_dir,
            stem="anchor_gap_pairs_normal",
            metadata=pair_meta,
        )
        if normal_csv is not None and normal_pkl is not None:
            logging.info(f"Anchor-gap pair records (normal/all) CSV -> {normal_csv}")
            logging.info(f"Anchor-gap pair records (normal/all) PKL -> {normal_pkl}")

        intersection_records = [r for r in pair_records_all if bool(r.get("is_intersection", False))]
        if len(intersection_records) > 0:
            inter_csv, inter_pkl = _save_pair_records(
                intersection_records,
                args.vis_dir,
                stem="anchor_gap_pairs_intersection",
                metadata=pair_meta,
            )
            if inter_csv is not None and inter_pkl is not None:
                logging.info(f"Anchor-gap pair records (intersection) CSV -> {inter_csv}")
                logging.info(f"Anchor-gap pair records (intersection) PKL -> {inter_pkl}")
        else:
            logging.info("Anchor-gap pair records: no intersection rows to export for this run.")

    if anchor_gap_hist is not None:
        hist_csv, hist_png = _save_anchor_gap_histogram(anchor_gap_hist, args.vis_dir)
        if hist_csv is not None and hist_png is not None:
            logging.info(f"Anchor-gap histogram CSV -> {hist_csv}")
            logging.info(f"Anchor-gap histogram PNG -> {hist_png}")
        if args.anchor_selection == "uniform_gap":
            logging.info(f"Global anchor-gap counts -> {_format_hist_counts(anchor_gap_hist)}")


def parse_args():
    p = argparse.ArgumentParser(
        description="BIWI evaluation with square MTCNN crops (and optional native TokenHPE/TRG protocols)"
    )

    p.add_argument("--biwi_dir", default="/gpu-data4/filby/head_pose_datasets/faces_0")
    p.add_argument(
        "--test_subjects",
        type=int,
        nargs="+",
        default=list(range(1, 25)),
        help="Subject IDs to evaluate (default: all 24 BIWI subjects).",
    )
    p.add_argument("--crop_size", type=int, default=256)
    p.add_argument(
        "--pair_csv",
        default="",
        help=(
            "Optional explicit anchor/query pair CSV with columns "
            "subject, anchor_frame_num, anchor_pose_file, query_frame_num, query_pose_file. "
            "When set, the evaluator uses only these pairs instead of the built-in anchor selection."
        ),
    )

    p.add_argument("--base_checkpoint", default=os.path.join(base.SCRIPT_DIR, "checkpoints", "VGGT-1B", "model.pt"))
    p.add_argument(
        "--lora_checkpoint",
        default=os.path.join(base.SCRIPT_DIR, "training", "logs", "flame_h2c_lora", "ckpts", "checkpoint.pt"),
    )
    p.add_argument("--no_lora", action="store_true")
    p.add_argument(
        "--vggt_pose_mode",
        choices=["relative_h2c", "absolute_second", "absolute_single"],
        default="relative_h2c",
        help="How to decode the target-frame VGGT prediction during BIWI evaluation.",
    )
    p.add_argument(
        "--vggt_prediction_head_yz_flip",
        action="store_true",
        help=(
            "Right-multiply VGGT predicted rotations by diag(1,-1,-1) before scoring. "
            "Use this only when the dataset GT has been adapted with the FaMoS head_yz_flip convention."
        ),
    )
    p.add_argument(
        "--anchor_selection",
        choices=["fixed_first", "random_far", "uniform_gap", "close_pose", "autoregressive"],
        default="fixed_first",
        help=(
            "Anchor selection for 2-view VGGT modes. "
            "`fixed_first`: frame 0 anchor (default). "
            "`random_far`: random anchor per target constrained by --min_anchor_frame_gap. "
            "`uniform_gap`: picks anchors to flatten anchor-target GT rotation-gap bins. "
            "`close_pose`: picks an anchor with GT rotation gap <= --max_anchor_pose_gap_deg. "
            "`autoregressive`: next anchor pose is previous VGGT prediction (sequential chain)."
        ),
    )
    p.add_argument(
        "--min_anchor_frame_gap",
        type=int,
        default=50,
        help="Minimum absolute frame-number distance |anchor-target| for random_far/uniform_gap modes.",
    )
    p.add_argument(
        "--anchor_seed",
        type=int,
        default=12345,
        help="Seed for random_far/uniform_gap/close_pose anchor sampling. Set <0 for non-deterministic sampling.",
    )
    p.add_argument(
        "--max_anchor_pose_gap_deg",
        type=float,
        default=15.0,
        help="For close_pose mode: maximum allowed GT geodesic anchor-target rotation gap in degrees.",
    )
    p.add_argument(
        "--close_anchor_min_frame_gap",
        type=int,
        default=1,
        help="For close_pose mode: minimum absolute frame-number distance |anchor-target|.",
    )
    p.add_argument(
        "--anchor_gap_bins_deg",
        type=float,
        nargs="+",
        default=[0, 5, 10, 15, 20, 30, 45, 60, 90, 180],
        help=(
            "Bin edges (deg) for histogram of model error vs GT anchor-target rotation gap "
            "(used in 2-view modes)."
        ),
    )
    p.add_argument(
        "--vggt_translation_scale",
        type=float,
        default=1000.0,
        help=(
            "Multiplier applied to decoded VGGT translation before BIWI comparison "
            "(default 1000.0 assumes model outputs meters and BIWI is in mm)."
        ),
    )
    p.add_argument("--enable_track", action="store_true", help="Load pretrained track head (no query_points -> dormant at eval)")

    p.add_argument("--sixdrepnet_dir", default=os.path.join(base.SCRIPT_DIR, "..", "6DRepNet", "sixdrepnet"))
    p.add_argument(
        "--sixdrepnet_checkpoint",
        default=os.path.join(base.SCRIPT_DIR, "..", "6DRepNet", "sixdrepnet", "6DRepNet_70_30_BIWI.pth"),
    )
    p.add_argument("--no_sixdrepnet", action="store_true")
    p.add_argument("--tokenhpe_repo", default="/leonardo_work/EUHPC_D32_089/head_pose/HPE_Benchmarking/TokenHPE")
    p.add_argument(
        "--tokenhpe_checkpoint",
        default=os.path.join(
            "/leonardo_work/EUHPC_D32_089/head_pose/HPE_Benchmarking/TokenHPE",
            "checkpoints",
            "TokenHPEv1-ViTB-224_224-lyr3.tar",
        ),
    )
    p.add_argument("--no_tokenhpe", action="store_true")
    p.add_argument(
        "--tokenhpe_native_npz",
        default="/leonardo_work/EUHPC_D32_089/head_pose/HPE_Benchmarking/6DRepNet/BIWI_test.npz",
        help="Native TokenHPE BIWI npz file used by official TokenHPE test protocol.",
    )
    p.add_argument("--tokenhpe_native_batch_size", type=int, default=64)
    p.add_argument("--tokenhpe_native_num_workers", type=int, default=2)
    p.add_argument("--whenet_repo", default="/leonardo_work/EUHPC_D32_089/head_pose/HeadPoseEstimation-WHENet")
    p.add_argument(
        "--whenet_checkpoint",
        default=os.path.join(
            "/leonardo_work/EUHPC_D32_089/head_pose/HeadPoseEstimation-WHENet",
            "WHENet.h5",
        ),
    )
    p.add_argument("--no_whenet", action="store_true")
    p.add_argument("--trg_repo", default="/leonardo_work/EUHPC_D32_089/head_pose/HPE_Benchmarking/TRG-Release")
    p.add_argument(
        "--trg_checkpoint",
        default="",
        help="Direct TRG checkpoint file path (state_dict.bin). If empty, resolve from --trg_checkpoints_dir/--trg_name/--trg_epoch.",
    )
    p.add_argument(
        "--trg_checkpoints_dir",
        default="",
        help="Root checkpoints directory for TRG runs (default: <trg_repo>/checkpoint).",
    )
    p.add_argument("--trg_name", default="trg_single_240717")
    p.add_argument("--trg_epoch", default="30")
    p.add_argument("--trg_img_size", type=int, default=192)
    p.add_argument("--trg_native_batch_size", type=int, default=64)
    p.add_argument("--trg_native_num_workers", type=int, default=4)
    p.add_argument(
        "--trg_native_csv_path",
        default="dataset/ARKitFace/ARKitFace_list/list/ARKitFace_test.csv",
        help="Placeholder csv_path_test argument passed to official TRG test.py in native protocol mode.",
    )
    p.add_argument(
        "--trg_native_annot",
        default="",
        help="Optional explicit path to TRG BIWI annot_mtcnn_fan.pkl for native protocol checks.",
    )
    p.add_argument(
        "--trg_shared_crop_source",
        choices=["mtcnn", "trg_fan"],
        default="mtcnn",
        help=(
            "TRG crop source in shared comparison mode. "
            "'mtcnn' uses the shared MTCNN box; "
            "'trg_fan' uses TRG annot_mtcnn_fan.pkl FAN landmarks (native-style bbox)."
        ),
    )
    p.add_argument("--no_trg", action="store_true")
    p.add_argument(
        "--native_protocol",
        action="store_true",
        help=(
            "Use official/native evaluation protocols for TokenHPE and TRG "
            "(TokenHPE BIWI npz and TRG official test.py). "
            "Shared-crop per-frame TokenHPE/TRG inference is disabled in this mode."
        ),
    )
    p.add_argument(
        "--native_protocol_keep_shared",
        action="store_true",
        help=(
            "When --native_protocol is enabled, also keep shared-crop TokenHPE/TRG inference "
            "so both shared and native TRG summaries can be reported in one run."
        ),
    )

    p.add_argument("--sixdof_face_repo", default="/leonardo_work/EUHPC_D32_089/head_pose/HPE_Benchmarking/6dof_face")
    p.add_argument(
        "--sixdof_face_data_root",
        default="",
        help="Preprocessed 6DoF Face BIWI evaluation root (contains csv/, image/, info/). Required if not under sixdof_face_repo/dataset/ARKitFace.",
    )
    p.add_argument("--sixdof_face_checkpoints_dir", default="")
    p.add_argument(
        "--sixdof_face_cache_root",
        default=os.path.join(base.SCRIPT_DIR, "cache", "sixdof_face_biwi_eval"),
        help="Where to auto-build the 6DoF Face BIWI evaluation root if it is missing.",
    )
    p.add_argument(
        "--prepare_sixdof_face_if_missing",
        action="store_true",
        help="Auto-build the 6DoF Face BIWI evaluation root from --biwi_dir when it is missing.",
    )
    p.add_argument("--sixdof_face_name", default="run1")
    p.add_argument("--sixdof_face_epoch", default="latest")
    p.add_argument("--sixdof_face_model", default="perspnet")
    p.add_argument("--sixdof_face_img_size", type=int, default=192)
    p.add_argument("--sixdof_face_batch_size", type=int, default=64)
    p.add_argument("--sixdof_face_num_workers", type=int, default=4)
    p.add_argument("--no_sixdof_face", action="store_true")

    p.add_argument(
        "--face_detector",
        choices=["mtcnn", "yunet"],
        default="mtcnn",
        help="Face detector used to define the shared crop for every active model.",
    )
    p.add_argument(
        "--yunet_model",
        default=os.path.join(
            base.SCRIPT_DIR,
            "revision_runs",
            "detector_sensitivity",
            "models",
            "face_detection_yunet_2023mar.onnx",
        ),
    )
    p.add_argument("--yunet_score_threshold", type=float, default=0.9)
    p.add_argument("--yunet_nms_threshold", type=float, default=0.3)
    p.add_argument("--mtcnn_ad", type=float, default=0.4, help="Shared detector-box enlargement margin")

    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--gpu", type=int, default=0)

    p.add_argument("--visualize", action="store_true")
    p.add_argument("--vis_dir", default=os.path.join(base.SCRIPT_DIR, "vis_output_mtcnn_square_compare"))
    p.add_argument("--vis_every_n", type=int, default=1)

    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
