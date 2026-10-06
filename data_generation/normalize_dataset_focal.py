"""
Create a focal-normalized copy of a rendered FLAME dataset.

Run with Blender's Python so numpy/PIL are available in this environment:

    ./blender-3.6.23-linux-x64/blender --background --python normalize_dataset_focal.py -- \
      --src sample_outputs/dataset_20k_hair_v2 \
      --dst sample_outputs/dataset_20k_hair_v2_fixed_focal \
      --target_mode median

The script preserves the existing hierarchy and pose files, but rewrites:
  - output.png with an image-plane focal normalization warp
  - output_opencv_camera.pkl with the target fx/fy
  - face_bbox.npy with the same image-plane transform

Unchanged files are hardlinked when possible to avoid duplicating the full
dataset. If hardlinking fails, files are copied.
"""

from __future__ import annotations

import argparse
import os
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args():
    argv = sys.argv
    argv = argv[argv.index("--") + 1 :] if "--" in argv else []
    parser = argparse.ArgumentParser(description="Normalize dataset focal length.")
    parser.add_argument("--src", required=True, help="Source dataset root.")
    parser.add_argument("--dst", required=True, help="Destination dataset root.")
    parser.add_argument(
        "--target_mode",
        choices=["max", "median", "mean", "first", "manual"],
        default="max",
        help="How to choose the common target focal length.",
    )
    parser.add_argument("--target_fx", type=float, default=None)
    parser.add_argument("--target_fy", type=float, default=None)
    parser.add_argument("--overwrite", type=int, default=0)
    parser.add_argument("--dry_run", type=int, default=0)
    parser.add_argument(
        "--copy_mode",
        choices=["hardlink", "copy"],
        default="hardlink",
        help="Use hardlinks for unchanged files when possible.",
    )
    parser.add_argument(
        "--fill",
        default="0,0,0",
        help="RGB fill color for areas outside the source image after warping.",
    )
    parser.add_argument(
        "--write_view_json",
        type=int,
        default=1,
        help="Write focal_normalization.json in every rewritten view directory.",
    )
    return parser.parse_args(argv)


def load_camera(path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def save_pickle(path, value):
    with open(path, "wb") as handle:
        pickle.dump(value, handle)


def iter_view_dirs(root):
    root = Path(root)
    for path in sorted(root.glob("identity_*/lighting_*/expr_*/view_*")):
        if (path / "output_opencv_camera.pkl").exists() and (path / "output.png").exists():
            yield path


def collect_focals(view_dirs):
    focals = []
    for view_dir in view_dirs:
        cam = load_camera(view_dir / "output_opencv_camera.pkl")
        K = np.asarray(cam["K"], dtype=np.float64).reshape(3, 3)
        focals.append([float(K[0, 0]), float(K[1, 1])])
    if not focals:
        raise RuntimeError("No view directories with output_opencv_camera.pkl and output.png found.")
    return np.asarray(focals, dtype=np.float64)


def choose_target_focal(focals, args):
    if args.target_mode == "manual":
        if args.target_fx is None or args.target_fy is None:
            raise ValueError("--target_mode manual requires --target_fx and --target_fy")
        return float(args.target_fx), float(args.target_fy)
    if args.target_mode == "max":
        return float(np.max(focals[:, 0])), float(np.max(focals[:, 1]))
    if args.target_mode == "first":
        return float(focals[0, 0]), float(focals[0, 1])
    if args.target_mode == "mean":
        return float(np.mean(focals[:, 0])), float(np.mean(focals[:, 1]))
    return float(np.median(focals[:, 0])), float(np.median(focals[:, 1]))


def link_or_copy(src, dst, copy_mode):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    if copy_mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def copy_tree_with_rewrites(src_root, dst_root, rewrite_relpaths, copy_mode, overwrite, dry_run):
    if dst_root.exists():
        if not overwrite:
            raise FileExistsError(f"{dst_root} exists; pass --overwrite 1 to replace files in it")
    elif not dry_run:
        dst_root.mkdir(parents=True)

    for src in src_root.rglob("*"):
        rel = src.relative_to(src_root)
        dst = dst_root / rel
        if src.is_dir():
            if not dry_run:
                dst.mkdir(parents=True, exist_ok=True)
            continue
        if rel in rewrite_relpaths:
            continue
        if not dry_run:
            link_or_copy(src, dst, copy_mode)


def parse_fill(fill_text, mode):
    values = tuple(int(v.strip()) for v in fill_text.split(",") if v.strip())
    if mode == "RGBA":
        if len(values) == 3:
            return values + (255,)
        if len(values) == 4:
            return values
    if len(values) == 1:
        return values[0]
    if len(values) >= 3:
        return values[:3]
    return 0


def warp_image_to_target_focal(image_path, dst_path, K_old, target_fx, target_fy, fill_text):
    image = Image.open(image_path)
    width, height = image.size
    fx_old = float(K_old[0, 0])
    fy_old = float(K_old[1, 1])
    cx_old = float(K_old[0, 2])
    cy_old = float(K_old[1, 2])
    sx = target_fx / fx_old
    sy = target_fy / fy_old

    # Output -> input affine mapping.
    coeffs = (
        1.0 / sx,
        0.0,
        cx_old - cx_old / sx,
        0.0,
        1.0 / sy,
        cy_old - cy_old / sy,
    )
    fill = parse_fill(fill_text, image.mode)
    warped = image.transform(
        (width, height),
        Image.Transform.AFFINE,
        coeffs,
        resample=Image.Resampling.BICUBIC,
        fillcolor=fill,
    )
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    warped.save(dst_path)


def transform_bbox(bbox, K_old, target_fx, target_fy, width, height):
    xmin, xmax, ymin, ymax = [float(v) for v in bbox]
    fx_old = float(K_old[0, 0])
    fy_old = float(K_old[1, 1])
    cx = float(K_old[0, 2])
    cy = float(K_old[1, 2])
    sx = target_fx / fx_old
    sy = target_fy / fy_old

    xs = [xmin, xmax]
    ys = [ymin, ymax]
    xs_new = [cx + sx * (x - cx) for x in xs]
    ys_new = [cy + sy * (y - cy) for y in ys]
    xmin_new = float(np.clip(min(xs_new), 0, width - 1))
    xmax_new = float(np.clip(max(xs_new), 0, width - 1))
    ymin_new = float(np.clip(min(ys_new), 0, height - 1))
    ymax_new = float(np.clip(max(ys_new), 0, height - 1))
    return np.asarray([xmin_new, xmax_new, ymin_new, ymax_new], dtype=np.float32)


def update_camera_focal(cam, target_fx, target_fy):
    cam = dict(cam)
    K = np.asarray(cam["K"], dtype=np.float64).reshape(3, 3).copy()
    K[0, 0] = target_fx
    K[1, 1] = target_fy
    cam["K"] = K
    if "RT" in cam:
        cam["P"] = K @ np.asarray(cam["RT"], dtype=np.float64)
    return cam


def write_root_manifest(dst_root, src_root, args, focals, target_fx, target_fy, view_count):
    manifest = {
        "source": str(src_root),
        "target_mode": args.target_mode,
        "target_fx": float(target_fx),
        "target_fy": float(target_fy),
        "view_count": int(view_count),
        "source_fx_min": float(np.min(focals[:, 0])),
        "source_fx_max": float(np.max(focals[:, 0])),
        "source_fx_mean": float(np.mean(focals[:, 0])),
        "source_fx_median": float(np.median(focals[:, 0])),
        "source_fy_min": float(np.min(focals[:, 1])),
        "source_fy_max": float(np.max(focals[:, 1])),
        "source_fy_mean": float(np.mean(focals[:, 1])),
        "source_fy_median": float(np.median(focals[:, 1])),
        "image_warp": "scale_about_principal_point",
        "pose_changed": False,
        "updated_files_per_view": [
            "output.png",
            "output_opencv_camera.pkl",
            "face_bbox.npy",
        ],
    }
    import json

    with open(dst_root / "focal_normalization_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


def write_view_manifest(dst_view, rel_dir, K_old, target_fx, target_fy):
    import json

    fx_old = float(K_old[0, 0])
    fy_old = float(K_old[1, 1])
    manifest = {
        "view": str(rel_dir),
        "old_fx": fx_old,
        "old_fy": fy_old,
        "new_fx": float(target_fx),
        "new_fy": float(target_fy),
        "scale_x": float(target_fx / fx_old),
        "scale_y": float(target_fy / fy_old),
        "principal_point_px": [float(K_old[0, 2]), float(K_old[1, 2])],
        "image_warp": "scale_about_principal_point",
        "pose_changed": False,
    }
    with open(dst_view / "focal_normalization.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


def main():
    args = parse_args()
    src_root = Path(args.src)
    dst_root = Path(args.dst)
    if not src_root.exists():
        raise FileNotFoundError(src_root)

    view_dirs = list(iter_view_dirs(src_root))
    focals = collect_focals(view_dirs)
    target_fx, target_fy = choose_target_focal(focals, args)

    rewrite_relpaths = set()
    for view_dir in view_dirs:
        for name in ("output.png", "output_opencv_camera.pkl", "face_bbox.npy"):
            rewrite_relpaths.add((view_dir / name).relative_to(src_root))

    print(f"Source: {src_root}")
    print(f"Destination: {dst_root}")
    print(f"Views: {len(view_dirs)}")
    print(
        "fx min/max/median/target: "
        f"{focals[:,0].min():.3f} / {focals[:,0].max():.3f} / "
        f"{np.median(focals[:,0]):.3f} / {target_fx:.3f}"
    )
    print(
        "fy min/max/median/target: "
        f"{focals[:,1].min():.3f} / {focals[:,1].max():.3f} / "
        f"{np.median(focals[:,1]):.3f} / {target_fy:.3f}"
    )

    copy_tree_with_rewrites(
        src_root,
        dst_root,
        rewrite_relpaths,
        copy_mode=args.copy_mode,
        overwrite=bool(args.overwrite),
        dry_run=bool(args.dry_run),
    )

    if args.dry_run:
        print("Dry run complete; no files written.")
        return

    write_root_manifest(dst_root, src_root, args, focals, target_fx, target_fy, len(view_dirs))

    for idx, view_dir in enumerate(view_dirs, start=1):
        rel_dir = view_dir.relative_to(src_root)
        dst_view = dst_root / rel_dir
        cam = load_camera(view_dir / "output_opencv_camera.pkl")
        K_old = np.asarray(cam["K"], dtype=np.float64).reshape(3, 3)
        width = int(cam.get("image_width", Image.open(view_dir / "output.png").size[0]))
        height = int(cam.get("image_height", Image.open(view_dir / "output.png").size[1]))

        warp_image_to_target_focal(
            view_dir / "output.png",
            dst_view / "output.png",
            K_old,
            target_fx,
            target_fy,
            args.fill,
        )
        save_pickle(
            dst_view / "output_opencv_camera.pkl",
            update_camera_focal(cam, target_fx, target_fy),
        )
        bbox_path = view_dir / "face_bbox.npy"
        if bbox_path.exists():
            bbox = np.load(bbox_path)
            np.save(
                dst_view / "face_bbox.npy",
                transform_bbox(bbox, K_old, target_fx, target_fy, width, height),
            )
        if args.write_view_json:
            write_view_manifest(dst_view, rel_dir, K_old, target_fx, target_fy)

        if idx % 1000 == 0:
            print(f"Processed {idx}/{len(view_dirs)} views")

    print("Focal-normalized dataset copy complete.")


if __name__ == "__main__":
    main()
