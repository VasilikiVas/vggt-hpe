#!/usr/bin/env python3
"""
Check a focal-normalized dataset copy with mesh projection overlays.

Run with Blender's Python in this environment:

    ./blender-3.6.23-linux-x64/blender --background --python check_fixed_focal_overlay.py -- \
      --src sample_outputs/dataset_20k_hair_v2 \
      --fixed sample_outputs/dataset_20k_hair_v2_fixed_focal_max \
      --out sample_outputs/fixed_focal_overlay_check

The test compares each selected original/fixed view pair:
  - old projection on original image
  - fixed projection on fixed image
  - old projection on fixed image, as a deliberate wrong-K reference

It also reports numeric residuals for the image warp, bbox warp, and vertex
projection warp induced by K_new K_old^{-1}.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFont


def parse_args():
    argv = sys.argv
    argv = argv[argv.index("--") + 1 :] if "--" in argv else []
    parser = argparse.ArgumentParser(description="Overlay-check fixed-focal dataset copy.")
    parser.add_argument("--src", required=True, help="Original dataset root.")
    parser.add_argument("--fixed", required=True, help="Focal-normalized dataset root.")
    parser.add_argument("--out", required=True, help="Directory for overlay images and summary JSON.")
    parser.add_argument("--num_random", type=int, default=3, help="Extra deterministic samples.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--point_stride", type=int, default=3)
    parser.add_argument("--point_radius", type=int, default=1)
    parser.add_argument("--max_views_scan", type=int, default=0, help="0 means scan all fixed views.")
    return parser.parse_args(argv)


def load_pickle(path: Path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def load_camera(view_dir: Path):
    cam = load_pickle(view_dir / "output_opencv_camera.pkl")
    K = np.asarray(cam["K"], dtype=np.float64).reshape(3, 3)
    R = np.asarray(cam["R"], dtype=np.float64).reshape(3, 3)
    t = np.asarray(cam["t"], dtype=np.float64).reshape(3)
    h = int(cam.get("image_height", 1024))
    w = int(cam.get("image_width", 1024))
    return K, R, t, h, w


def load_h2c(view_dir: Path):
    K, R_w2c, t_w2c, h, w = load_camera(view_dir)
    obj = load_pickle(view_dir / "output_object_transform_post_render.pkl")
    R_h2w = np.asarray(obj["R_world"], dtype=np.float64).reshape(3, 3)
    t_h2w = np.asarray(obj["t_world"], dtype=np.float64).reshape(3)
    R_h2c = R_w2c @ R_h2w
    t_h2c = R_w2c @ t_h2w + t_w2c
    return K, R_h2c, t_h2c, h, w


def load_obj_vertices(path: Path):
    verts = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("v "):
                parts = line.split()
                if len(parts) >= 4:
                    verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
    if not verts:
        raise RuntimeError(f"No OBJ vertices found in {path}")
    return np.asarray(verts, dtype=np.float64)


def project_vertices(vertices, K, R_h2c, t_h2c):
    cam = vertices @ R_h2c.T + t_h2c.reshape(1, 3)
    z = cam[:, 2]
    valid_z = z > 1e-6
    uv = np.full((vertices.shape[0], 2), np.nan, dtype=np.float64)
    uv[valid_z, 0] = K[0, 0] * (cam[valid_z, 0] / z[valid_z]) + K[0, 2]
    uv[valid_z, 1] = K[1, 1] * (cam[valid_z, 1] / z[valid_z]) + K[1, 2]
    return uv, valid_z


def inside_mask(uv, valid_z, w, h):
    return (
        valid_z
        & np.isfinite(uv[:, 0])
        & np.isfinite(uv[:, 1])
        & (uv[:, 0] >= 0)
        & (uv[:, 0] < w)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < h)
    )


def projected_bbox(uv, mask):
    pts = uv[mask]
    if pts.size == 0:
        return None
    return np.asarray(
        [pts[:, 0].min(), pts[:, 0].max(), pts[:, 1].min(), pts[:, 1].max()],
        dtype=np.float64,
    )


def warp_uv(uv, K_old, K_new):
    out = uv.copy()
    sx = K_new[0, 0] / K_old[0, 0]
    sy = K_new[1, 1] / K_old[1, 1]
    out[:, 0] = K_new[0, 2] + sx * (uv[:, 0] - K_old[0, 2])
    out[:, 1] = K_new[1, 2] + sy * (uv[:, 1] - K_old[1, 2])
    return out


def warp_bbox(bbox, K_old, K_new, w, h):
    sx = K_new[0, 0] / K_old[0, 0]
    sy = K_new[1, 1] / K_old[1, 1]
    cx_old, cy_old = K_old[0, 2], K_old[1, 2]
    cx_new, cy_new = K_new[0, 2], K_new[1, 2]
    xmin, xmax, ymin, ymax = [float(v) for v in bbox]
    xs = [cx_new + sx * (x - cx_old) for x in (xmin, xmax)]
    ys = [cy_new + sy * (y - cy_old) for y in (ymin, ymax)]
    return np.asarray(
        [
            np.clip(min(xs), 0, w - 1),
            np.clip(max(xs), 0, w - 1),
            np.clip(min(ys), 0, h - 1),
            np.clip(max(ys), 0, h - 1),
        ],
        dtype=np.float64,
    )


def warp_image_like_normalizer(image, K_old, K_new):
    width, height = image.size
    sx = K_new[0, 0] / K_old[0, 0]
    sy = K_new[1, 1] / K_old[1, 1]
    cx = K_old[0, 2]
    cy = K_old[1, 2]
    coeffs = (
        1.0 / sx,
        0.0,
        cx - cx / sx,
        0.0,
        1.0 / sy,
        cy - cy / sy,
    )
    return image.transform(
        (width, height),
        Image.Transform.AFFINE,
        coeffs,
        resample=Image.Resampling.BICUBIC,
        fillcolor=(0, 0, 0),
    )


def image_diff_stats(expected, actual):
    diff = np.asarray(ImageChops.difference(expected.convert("RGB"), actual.convert("RGB")), dtype=np.float32)
    return {
        "mean_abs_rgb_diff": float(diff.mean()),
        "max_abs_rgb_diff": float(diff.max()),
        "nonzero_pixels": int(np.count_nonzero(diff.max(axis=2))),
    }


def draw_overlay(image, uv, mask, bbox=None, color=(40, 255, 40), label=None, stride=3, radius=1):
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    idx = np.where(mask)[0][:: max(1, int(stride))]
    r = int(radius)
    for i in idx:
        x, y = float(uv[i, 0]), float(uv[i, 1])
        if r <= 1:
            draw.point((x, y), fill=color)
        else:
            draw.ellipse((x - r, y - r, x + r, y + r), fill=color)
    if bbox is not None:
        xmin, xmax, ymin, ymax = [float(v) for v in bbox]
        draw.rectangle((xmin, ymin, xmax, ymax), outline=(255, 255, 0), width=2)
    if label:
        draw.rectangle((0, 0, out.width, 34), fill=(0, 0, 0))
        draw.text((8, 8), label, fill=(255, 255, 255), font=ImageFont.load_default())
    return out


def select_views(fixed_root: Path, num_random: int, seed: int, max_views_scan: int):
    view_dirs = sorted(fixed_root.glob("identity_*/lighting_*/expr_*/view_*"))
    view_dirs = [p for p in view_dirs if (p / "focal_normalization.json").exists()]
    if max_views_scan and max_views_scan > 0:
        view_dirs = view_dirs[:max_views_scan]
    records = []
    for view_dir in view_dirs:
        meta = json.loads((view_dir / "focal_normalization.json").read_text(encoding="utf-8"))
        records.append((float(meta["scale_x"]), view_dir))
    if not records:
        raise RuntimeError(f"No focal_normalization.json files found under {fixed_root}")
    records.sort(key=lambda item: item[0])
    picks = [records[0], records[len(records) // 2], records[-1]]
    rng = np.random.default_rng(seed)
    if num_random > 0:
        for idx in rng.choice(len(records), size=min(num_random, len(records)), replace=False):
            picks.append(records[int(idx)])
    unique = []
    seen = set()
    for scale, view_dir in picks:
        rel = view_dir.relative_to(fixed_root)
        if rel not in seen:
            seen.add(rel)
            unique.append((scale, view_dir))
    return unique


def process_view(src_root: Path, fixed_root: Path, fixed_view: Path, out_dir: Path, label: str, args):
    rel = fixed_view.relative_to(fixed_root)
    src_view = src_root / rel

    K_old, R_old, t_old, h, w = load_h2c(src_view)
    K_new, R_new, t_new, h_new, w_new = load_h2c(fixed_view)

    vertices = load_obj_vertices(src_view / "output_mesh.obj")
    uv_old, valid_old = project_vertices(vertices, K_old, R_old, t_old)
    uv_new, valid_new = project_vertices(vertices, K_new, R_new, t_new)
    mask_old = inside_mask(uv_old, valid_old, w, h)
    mask_new = inside_mask(uv_new, valid_new, w_new, h_new)

    uv_old_warped = warp_uv(uv_old, K_old, K_new)
    common = valid_old & valid_new & np.isfinite(uv_old_warped[:, 0]) & np.isfinite(uv_new[:, 0])
    uv_residual = uv_new[common] - uv_old_warped[common]
    uv_rms = float(np.sqrt(np.mean(np.sum(uv_residual * uv_residual, axis=1)))) if uv_residual.size else None
    uv_max = float(np.max(np.abs(uv_residual))) if uv_residual.size else None

    bbox_old = np.load(src_view / "face_bbox.npy").astype(np.float64)
    bbox_new = np.load(fixed_view / "face_bbox.npy").astype(np.float64)
    bbox_expected = warp_bbox(bbox_old, K_old, K_new, w_new, h_new)
    bbox_max_abs = float(np.max(np.abs(bbox_new - bbox_expected)))

    orig_img = Image.open(src_view / "output.png").convert("RGB")
    fixed_img = Image.open(fixed_view / "output.png").convert("RGB")
    expected_fixed = warp_image_like_normalizer(orig_img, K_old, K_new)
    diff_stats = image_diff_stats(expected_fixed, fixed_img)

    old_proj_bbox = projected_bbox(uv_old, mask_old)
    new_proj_bbox = projected_bbox(uv_new, mask_new)

    old_overlay = draw_overlay(
        orig_img,
        uv_old,
        mask_old,
        bbox=bbox_old,
        color=(40, 255, 40),
        label=f"original + original K  fx={K_old[0,0]:.1f}",
        stride=args.point_stride,
        radius=args.point_radius,
    )
    fixed_overlay = draw_overlay(
        fixed_img,
        uv_new,
        mask_new,
        bbox=bbox_new,
        color=(40, 255, 40),
        label=f"fixed + fixed K  fx={K_new[0,0]:.1f}",
        stride=args.point_stride,
        radius=args.point_radius,
    )
    wrong_overlay = draw_overlay(
        fixed_img,
        uv_old,
        mask_old,
        bbox=bbox_old,
        color=(255, 60, 60),
        label="fixed + old K reference",
        stride=args.point_stride,
        radius=args.point_radius,
    )

    mosaic = Image.new("RGB", (w * 3, h), (0, 0, 0))
    mosaic.paste(old_overlay, (0, 0))
    mosaic.paste(fixed_overlay, (w, 0))
    mosaic.paste(wrong_overlay, (w * 2, 0))
    out_path = out_dir / f"{label}.png"
    mosaic.save(out_path)

    return {
        "view": str(rel),
        "overlay": str(out_path),
        "old_fx": float(K_old[0, 0]),
        "new_fx": float(K_new[0, 0]),
        "scale_x": float(K_new[0, 0] / K_old[0, 0]),
        "old_inside_vertices": int(mask_old.sum()),
        "fixed_inside_vertices": int(mask_new.sum()),
        "uv_warp_rms_px": uv_rms,
        "uv_warp_max_abs_px": uv_max,
        "bbox_warp_max_abs_px": bbox_max_abs,
        "image_warp": diff_stats,
        "old_projected_bbox": None if old_proj_bbox is None else old_proj_bbox.tolist(),
        "fixed_projected_bbox": None if new_proj_bbox is None else new_proj_bbox.tolist(),
        "old_face_bbox": bbox_old.tolist(),
        "fixed_face_bbox": bbox_new.tolist(),
    }


def main():
    args = parse_args()
    src_root = Path(args.src)
    fixed_root = Path(args.fixed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    selected = select_views(fixed_root, args.num_random, args.seed, args.max_views_scan)
    results = []
    for i, (scale, fixed_view) in enumerate(selected):
        label = f"{i:02d}_scale_{scale:.4f}_{fixed_view.parent.parent.name}_{fixed_view.name}"
        print(f"[overlay] {label}: {fixed_view.relative_to(fixed_root)}")
        results.append(process_view(src_root, fixed_root, fixed_view, out_dir, label, args))

    summary = {
        "src": str(src_root),
        "fixed": str(fixed_root),
        "out": str(out_dir),
        "num_samples": len(results),
        "results": results,
    }
    summary_path = out_dir / "overlay_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[done] wrote {len(results)} overlays")
    print(f"[done] summary: {summary_path}")
    for item in results:
        print(
            "[check]",
            item["view"],
            f"scale={item['scale_x']:.6f}",
            f"uv_rms={item['uv_warp_rms_px']:.6g}",
            f"uv_max={item['uv_warp_max_abs_px']:.6g}",
            f"bbox_max={item['bbox_warp_max_abs_px']:.6g}",
            f"img_mean={item['image_warp']['mean_abs_rgb_diff']:.6g}",
            f"img_max={item['image_warp']['max_abs_rgb_diff']:.6g}",
        )


if __name__ == "__main__":
    main()
