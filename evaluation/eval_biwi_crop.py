"""
biwi_eval.py
------------
End-to-end BIWI evaluation for VGGT (+ optional 6DRepNet).

Protocol : 70/30 subject split (default test subjects: 5 6 9 14 16 17 20 24)
Metric   : Angular-wrapping MAE on Yaw / Pitch / Roll (degrees)
Visuals  : multi-panel per-frame PNG
           Mesh projected with perspective using per-subject .obj + rgb.cal
           GT translation used for all panels; only rotation is swapped per model.

FIX:  6DRepNet now uses MTCNN face-detection crops (matching how the npz
      training data was created) instead of 3D-projection crops.  VGGT keeps
      its own 3D-projection crop.  Euler conventions unchanged (were correct).
"""

import os, sys, io, gc, glob, re, csv, argparse, logging
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import PolyCollection
from PIL import Image
from torchvision import transforms as TF
from tqdm import tqdm

# ── Paths / defaults ──────────────────────────────────────────────────────────
SCRIPT_DIR            = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TEST_SUBJECTS = [5, 6, 9, 14, 16, 17, 20, 24]
RELATIVE_VARIANT_NAMES = ["B_no_flip", "E_h2c_correct"]
ABSOLUTE_SECOND_VARIANT_NAMES = ["ABS_second"]
ABSOLUTE_SINGLE_VARIANT_NAMES = ["ABS_single"]

S_CV2P3D = torch.diag(torch.tensor([-1.0, -1.0, 1.0]))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def get_variant_names(vggt_pose_mode):
    if vggt_pose_mode == "relative_h2c":
        return RELATIVE_VARIANT_NAMES
    if vggt_pose_mode == "absolute_second":
        return ABSOLUTE_SECOND_VARIANT_NAMES
    if vggt_pose_mode == "absolute_single":
        return ABSOLUTE_SINGLE_VARIANT_NAMES
    raise ValueError(f"Unknown vggt_pose_mode: {vggt_pose_mode}")


def get_primary_variant_name(vggt_pose_mode):
    if vggt_pose_mode == "relative_h2c":
        return "E_h2c_correct"
    if vggt_pose_mode == "absolute_second":
        return "ABS_second"
    if vggt_pose_mode == "absolute_single":
        return "ABS_single"
    raise ValueError(f"Unknown vggt_pose_mode: {vggt_pose_mode}")


# ═════════════════════════════════════════════════════════════════════════════
# 1.  BIWI DATA HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def get_subject_frames(subject_dir):
    """Return sorted [(frame_num, rgb_path, pose_path), ...] for one subject."""
    frames = []
    for rgb_path in sorted(glob.glob(os.path.join(subject_dir, "frame_*_rgb.png"))):
        m = re.search(r"frame_(\d+)_rgb\.png", os.path.basename(rgb_path))
        pose_path = rgb_path.replace("_rgb.png", "_pose.txt")
        if m and os.path.exists(pose_path):
            frames.append((int(m.group(1)), rgb_path, pose_path))
    return frames


def parse_pose_txt(path):
    """Parse BIWI frame_XXXXX_pose.txt → R (3×3), t (3,) in mm."""
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip()]
    R = np.array([[float(v) for v in lines[i].split()] for i in range(3)], dtype=np.float64)
    t = np.array([float(v) for v in lines[3].split()], dtype=np.float64)
    return R, t


def load_rgb_intrinsics(path):
    """Parse rgb.cal → K (3×3), R_cam (3×3), t_cam (3,)."""
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip()]
    K     = np.array([[float(x) for x in lines[i].split()] for i in range(3)])
    R_cam = np.array([[float(x) for x in lines[i].split()] for i in range(4, 7)])
    t_cam = np.array([float(x) for x in lines[7].split()])
    return K, R_cam, t_cam


def load_obj(path):
    """Parse per-subject .obj mesh → vertices (V,3), faces (F,3)."""
    vertices, faces = [], []
    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                vertices.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                toks = [int(x.split("/")[0]) - 1 for x in line.split()[1:]]
                if len(toks) == 3:
                    faces.append(toks)
                elif len(toks) == 4:
                    faces += [toks[:3], [toks[0], toks[2], toks[3]]]
    return np.array(vertices, dtype=np.float64), np.array(faces, dtype=np.int64)


def load_img_rgb(path):
    img = cv2.imread(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_img_bgr(path):
    """Load as BGR (for MTCNN which expects BGR→RGB internally, but
       mtcnn.detect_faces actually wants RGB, so we convert)."""
    return cv2.imread(path)


def crop_face(img_rgb, t, K, crop_size=256):
    """Crop face region using projected 3D head-centre (for VGGT)."""
    h, w = img_rgb.shape[:2]
    u = int(round(K[0, 0] * t[0] / t[2] + K[0, 2]))
    v = int(round(K[1, 1] * t[1] / t[2] + K[1, 2]))
    half = crop_size // 2
    x1, y1 = u - half, v - half
    x2, y2 = x1 + crop_size, y1 + crop_size
    x1c, y1c = max(0, x1), max(0, y1)
    x2c, y2c = min(w, x2), min(h, y2)
    patch = img_rgb[y1c:y2c, x1c:x2c]
    if patch.shape[0] < crop_size or patch.shape[1] < crop_size:
        canvas = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
        canvas[y1c - y1:y1c - y1 + patch.shape[0],
               x1c - x1:x1c - x1 + patch.shape[1]] = patch
        return canvas
    return patch


def crop_face_mtcnn(img_bgr, detector, prev_box, crop_size=256, ad=0.4):
    """
    MTCNN-based face crop matching the npz creation script.
    Uses facenet-pytorch MTCNN (pure PyTorch, no TensorFlow).

    Returns (crop_rgb, new_box).  new_box is [xw1,xw2,yw1,yw2] for fallback.
    If detection fails or jumps too far, uses prev_box.
    """
    img_h, img_w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    # facenet-pytorch returns boxes (N,4) as [x1,y1,x2,y2] and probs (N,)
    boxes, probs = detector.detect(pil_img)

    chosen_box = None

    if boxes is not None and len(boxes) > 0:
        dis_list = []
        XY = []
        for i_d in range(len(boxes)):
            if probs[i_d] > 0.95:
                x1, y1, x2, y2 = boxes[i_d]
                w = x2 - x1
                h = y2 - y1
                xw1 = max(int(x1 - ad * w), 0)
                yw1 = max(int(y1 - ad * h), 0)
                xw2 = min(int(x2 + ad * w), img_w - 1)
                yw2 = min(int(y2 + ad * h), img_h - 1)
                XY.append([xw1, xw2, yw1, yw2])
                dis = abs(xw1 - img_w * 2 / 3) + abs(yw1 - img_h * 2 / 3)
                dis_list.append(dis)

        if len(dis_list) > 0:
            min_id = np.argmin(dis_list)
            candidate = XY[min_id]
            # Check jump distance against previous frame
            if prev_box is None or abs(candidate[0] - prev_box[0]) < 80:
                chosen_box = candidate

    # Fallback to previous box
    if chosen_box is None:
        if prev_box is not None:
            chosen_box = prev_box
        else:
            # No detection ever — fall back to centre crop
            cx, cy = img_w // 2, img_h // 2
            half = crop_size // 2
            chosen_box = [cx - half, cx + half, cy - half, cy + half]

    xw1, xw2, yw1, yw2 = chosen_box
    crop = img_bgr[yw1:yw2 + 1, xw1:xw2 + 1, :]
    # Keep BGR (no conversion) so PIL.fromarray interprets channels as RGB,
    # matching how the BIWI npz training data was created (cv2 BGR stored as-is).
    crop_bgr = cv2.resize(crop, (crop_size, crop_size))
    return crop_bgr, chosen_box


# ═════════════════════════════════════════════════════════════════════════════
# 2.  EULER ANGLES  (BIWI convention)
# ═════════════════════════════════════════════════════════════════════════════

def biwi_euler_deg_np(R):
    """Single 3×3 numpy R → (yaw, pitch, roll) degrees."""
    R_t   = R.T
    yaw   = -np.arctan2(-R_t[2, 0], np.sqrt(R_t[2, 1]**2 + R_t[2, 2]**2)) * (180 / np.pi)
    pitch =  np.arctan2(R_t[2, 1], R_t[2, 2]) * (180 / np.pi)
    roll  = -np.arctan2(R_t[1, 0], R_t[0, 0]) * (180 / np.pi)
    return yaw, pitch, roll


def biwi_euler_deg_batch(R_batch):
    """Batched torch (N,3,3) → yaw, pitch, roll each (N,) degrees."""
    R_t   = R_batch.transpose(-1, -2)
    yaw   = -torch.atan2(-R_t[..., 2, 0],
                          torch.sqrt(R_t[..., 2, 1]**2 + R_t[..., 2, 2]**2)) * (180 / np.pi)
    pitch =  torch.atan2(R_t[..., 2, 1], R_t[..., 2, 2]) * (180 / np.pi)
    roll  = -torch.atan2(R_t[..., 1, 0], R_t[..., 0, 0]) * (180 / np.pi)
    return yaw, pitch, roll


# ═════════════════════════════════════════════════════════════════════════════
# 3.  METRICS
# ═════════════════════════════════════════════════════════════════════════════

def angular_error(gt_deg, pred_deg):
    """Min angular error with 360/180 wrapping (matches 6DRepNet test.py)."""
    return torch.min(torch.stack([
        torch.abs(gt_deg - pred_deg),
        torch.abs(pred_deg + 360 - gt_deg),
        torch.abs(pred_deg - 360 - gt_deg),
        torch.abs(pred_deg + 180 - gt_deg),
        torch.abs(pred_deg - 180 - gt_deg),
    ]), dim=0)[0]


def _empty_variant_acc(variant_names):
    return {vname: {"y": 0.0, "p": 0.0, "r": 0.0} for vname in variant_names}


# ═════════════════════════════════════════════════════════════════════════════
# 4.  MODEL LOADING
# ═════════════════════════════════════════════════════════════════════════════

LORA_TARGET_PATTERNS = [
    "frame_blocks.*.attn.qkv",  "frame_blocks.*.attn.proj",
    "frame_blocks.*.mlp.fc1",   "frame_blocks.*.mlp.fc2",
    "global_blocks.*.attn.qkv", "global_blocks.*.attn.proj",
    "global_blocks.*.mlp.fc1",  "global_blocks.*.mlp.fc2",
]


def _state_uses_split_pose_branch(state):
    return any("camera_head.pose_branch_rot." in key for key in state.keys())


def load_vggt_model(base_ckpt_path, lora_ckpt_path=None, device="cuda", enable_track=False):
    from vggt.models.vggt import VGGT

    lora_cfg = None
    if lora_ckpt_path is not None:
        lora_cfg = dict(enabled=True, rank=8, alpha=16, dropout=0.0,
                        freeze_base=True, target_patterns=LORA_TARGET_PATTERNS)

    lckpt = None
    lstate = None
    split_pose_branch = False
    if lora_ckpt_path is not None:
        logging.info(f"Inspecting LoRA: {lora_ckpt_path}")
        lckpt = torch.load(lora_ckpt_path, map_location="cpu", weights_only=False)
        lstate = lckpt["model"] if "model" in lckpt else lckpt
        split_pose_branch = _state_uses_split_pose_branch(lstate)
        logging.info(f"  Camera split pose branch: {split_pose_branch}")

    model = VGGT(enable_camera=True, enable_depth=False,
                 enable_point=False, enable_track=enable_track,
                 pose_encoding_type="absT_quaR_FoV",
                 camera_head_split_pose_branch=split_pose_branch,
                 lora=lora_cfg)

    logging.info(f"Loading VGGT base: {base_ckpt_path}")
    ckpt  = torch.load(base_ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["model"] if "model" in ckpt else ckpt
    miss, unex = model.load_state_dict(state, strict=False)
    logging.info(f"  Base — missing: {len(miss)}, unexpected: {len(unex)}")

    if lora_ckpt_path is not None:
        logging.info(f"Loading LoRA: {lora_ckpt_path}")
        miss, unex = model.load_state_dict(lstate, strict=False)
        logging.info(f"  LoRA — missing: {len(miss)}, unexpected: {len(unex)}")
        if "epoch" in lckpt:
            logging.info(f"  Checkpoint epoch: {lckpt['epoch']}")

    model.eval().to(device)
    return model


def load_sixdrepnet_model(ckpt_path, sixdrepnet_dir, device="cuda"):
    sys.path.insert(0, sixdrepnet_dir)
    from model import SixDRepNet

    model = SixDRepNet(backbone_name="RepVGG-B1g2",
                       backbone_file="", deploy=True, pretrained=False)
    logging.info(f"Loading 6DRepNet: {ckpt_path}")
    saved = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(saved.get("model_state_dict", saved))
    model.eval().to(device)
    return model


# ═════════════════════════════════════════════════════════════════════════════
# 5.  POSE DECODING
# ═════════════════════════════════════════════════════════════════════════════

def decode_vggt_pose(pose_enc, R_anchor_t, R_cam_t, pose_mode="relative_h2c"):
    """
    pose_enc  : (B, S, D)  absT_quaR_FoV
    R_anchor_t: (3, 3) R_pose of anchor frame from pose.txt, only used in relative mode
    R_cam_t   : (3, 3) depth→RGB rotation from rgb.cal

    Returns   : dict of {variant_name: R_pose (B,3,3)} in pose.txt space.
    """
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    extrinsics, _ = pose_encoding_to_extri_intri(
        pose_enc.float(),
        image_size_hw=(518, 518),
        pose_encoding_type="absT_quaR_FoV",
        build_intrinsics=False,
    )
    RcT = R_cam_t.T.unsqueeze(0)    # (1, 3, 3) R_cam.T

    if pose_mode == "relative_h2c":
        # (B, 3, 3) OpenCV relative H2C rotation.
        R_rel = extrinsics[:, 1, :3, :3].cpu()
        Ra = R_anchor_t.unsqueeze(0)   # (1, 3, 3) anchor pose in pose.txt space
        Rc = R_cam_t.unsqueeze(0)      # (1, 3, 3) R_cam (depth→RGB)
        return {
            "B_no_flip": R_rel @ Ra,
            "E_h2c_correct": RcT @ R_rel @ Rc @ Ra,
        }

    if pose_mode == "absolute_second":
        # The target prediction is absolute H2C for frame 2.
        # Training convention: R_h2c = R_cam @ R_pose, so R_pose = R_cam.T @ R_h2c.
        R_abs_h2c = extrinsics[:, 1, :3, :3].cpu()
        return {
            "ABS_second": RcT @ R_abs_h2c,
        }

    if pose_mode == "absolute_single":
        # Single-view absolute H2C prediction for the only frame in the batch item.
        R_abs_h2c = extrinsics[:, 0, :3, :3].cpu()
        return {
            "ABS_single": RcT @ R_abs_h2c,
        }

    raise ValueError(f"Unknown pose_mode: {pose_mode}")


def decode_sixdrepnet_pose(R_pred_sixd):
    """
    R_pred is in get_R(pitch, yaw, roll) space.
    compute_euler gives [x,y,z] = [pitch_biwi, yaw_biwi, roll_biwi].
    Also reconstruct R in pose.txt space for mesh visualisation.
    """
    sy       = torch.sqrt(R_pred_sixd[:, 0, 0]**2 + R_pred_sixd[:, 1, 0]**2)
    singular = (sy < 1e-6).float()
    x  = torch.atan2( R_pred_sixd[:, 2, 1],  R_pred_sixd[:, 2, 2])
    y  = torch.atan2(-R_pred_sixd[:, 2, 0],  sy)
    z  = torch.atan2( R_pred_sixd[:, 1, 0],  R_pred_sixd[:, 0, 0])
    xs = torch.atan2(-R_pred_sixd[:, 1, 2],  R_pred_sixd[:, 1, 1])
    ys = torch.atan2(-R_pred_sixd[:, 2, 0],  sy)
    pitch = (x * (1 - singular) + xs * singular) * (180 / np.pi)
    yaw   = (y * (1 - singular) + ys * singular) * (180 / np.pi)
    roll  = (z * (1 - singular)               ) * (180 / np.pi)

    N = R_pred_sixd.shape[0]
    R_biwi_batch = np.stack([
        euler_to_R_biwi(yaw[i].item(), pitch[i].item(), roll[i].item())
        for i in range(N)
    ])
    return yaw, pitch, roll, R_biwi_batch


def euler_to_R_biwi(yaw_deg, pitch_deg, roll_deg):
    """Invert BIWI euler convention → R in pose.txt space (for mesh vis)."""
    r = -roll_deg  * (np.pi / 180.0)
    y = -yaw_deg   * (np.pi / 180.0)
    p =  pitch_deg * (np.pi / 180.0)
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(p), -np.sin(p)],
                   [0, np.sin(p),  np.cos(p)]])
    Ry = np.array([[ np.cos(y), 0, np.sin(y)],
                   [0,          1, 0         ],
                   [-np.sin(y), 0, np.cos(y)]])
    Rz = np.array([[np.cos(r), -np.sin(r), 0],
                   [np.sin(r),  np.cos(r), 0],
                   [0,          0,         1]])
    return (Rz @ Ry @ Rx).T


# ═════════════════════════════════════════════════════════════════════════════
# 6.  PERSPECTIVE-PROJECTION MESH RENDERER
# ═════════════════════════════════════════════════════════════════════════════

def _project_vertices(vertices, R, t, R_cam, t_cam, K):
    v_world = (R @ vertices.T).T + t
    v_cam   = (R_cam @ v_world.T).T + t_cam
    valid   = v_cam[:, 2] > 0
    v_img   = (K @ v_cam.T).T
    v_img  /= v_img[:, 2:3]
    return v_cam, v_img[:, :2], valid


def render_mesh_on_image(img_rgb, vertices, faces, R, t, R_cam, t_cam, K,
                          color="#4a90d9", alpha=0.5):
    h, w = img_rgb.shape[:2]
    v_cam, v_img, valid = _project_vertices(vertices, R, t, R_cam, t_cam, K)

    face_list, face_depths, face_normals_z = [], [], []
    for fi in faces:
        if not all(valid[fi]):
            continue
        pts = v_img[fi]
        e1, e2 = pts[1] - pts[0], pts[2] - pts[0]
        if (e1[0]*e2[1] - e1[1]*e2[0]) > 0:
            continue
        v0, v1, v2 = v_cam[fi]
        n  = np.cross(v1 - v0, v2 - v0)
        nm = np.linalg.norm(n)
        nz = n[2] / nm if nm > 1e-8 else 0
        face_list.append(pts)
        face_depths.append(v_cam[fi, 2].mean())
        face_normals_z.append(abs(nz))

    if not face_list:
        return img_rgb

    order    = np.argsort(face_depths)[::-1]
    base_rgb = np.array(mcolors.to_rgb(color))

    fig, ax = plt.subplots(figsize=(w / 100, h / 100), dpi=100)
    fig.patch.set_alpha(0)
    ax.set_facecolor((0, 0, 0, 0))

    polys  = [face_list[i] for i in order]
    colors = [(*base_rgb * max(0.3, face_normals_z[i]), alpha) for i in order]
    pc = PolyCollection(polys, facecolors=colors, edgecolors="none",
                        linewidths=0, antialiased=False)
    ax.add_collection(pc)
    ax.set_xlim(0, w); ax.set_ylim(h, 0)
    ax.set_aspect("equal"); ax.axis("off")
    plt.subplots_adjust(0, 0, 1, 1)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, transparent=True,
                bbox_inches="tight", pad_inches=0)
    plt.close(fig); buf.seek(0)
    overlay = np.array(Image.open(buf).resize((w, h), Image.BICUBIC))

    fa   = overlay[..., 3:].astype(np.float32) / 255.0
    comp = (img_rgb.astype(np.float32) * (1 - fa) +
            overlay[..., :3].astype(np.float32) * fa)
    return comp.clip(0, 255).astype(np.uint8)


# ═════════════════════════════════════════════════════════════════════════════
# 7.  VISUALIZATION
# ═════════════════════════════════════════════════════════════════════════════

def save_vis_frame(
    img_rgb, vertices, faces, K_rgb, R_cam, t_cam,
    R_gt, t_gt, euler_gt,
    vggt_variants,
    R_sixd, euler_sixd,
    save_path,
    prefix_panels=None,
    R_tokenhpe=None, euler_tokenhpe=None,
    R_whenet=None, euler_whenet=None,
    R_trg=None, euler_trg=None,
):
    def _title(name, euler):
        y, p, r = euler
        return f"{name}\nY:{y:.1f}  P:{p:.1f}  R:{r:.1f}"

    VARIANT_COLORS = {
        "A_full_flip" : "#e8744f",
        "B_no_flip"   : "#c0392b",
        "C_flip_Y"    : "#8e44ad",
        "D_order_flip": "#795548",
    }

    panel_gt = render_mesh_on_image(img_rgb.copy(), vertices, faces,
                                     R_gt, t_gt, R_cam, t_cam, K_rgb,
                                     color="#4a90d9", alpha=0.5)
    panels = []
    titles = []
    if prefix_panels:
        for panel_img, panel_title in prefix_panels:
            panels.append(panel_img)
            titles.append(panel_title)
    panels.extend([img_rgb, panel_gt])
    titles.extend(["RGB", _title("GT", euler_gt)])

    for vname, (R_v, euler_v) in vggt_variants.items():
        p = render_mesh_on_image(img_rgb.copy(), vertices, faces,
                                  R_v, t_gt, R_cam, t_cam, K_rgb,
                                  color=VARIANT_COLORS.get(vname, "#e8744f"),
                                  alpha=0.5)
        panels.append(p)
        titles.append(_title(vname, euler_v))

    if R_sixd is not None:
        panel_sixd = render_mesh_on_image(img_rgb.copy(), vertices, faces,
                                           R_sixd, t_gt, R_cam, t_cam, K_rgb,
                                           color="#50b86c", alpha=0.5)
        panels.append(panel_sixd)
        titles.append(_title("6DRepNet", euler_sixd))

    if R_tokenhpe is not None:
        panel_tokenhpe = render_mesh_on_image(
            img_rgb.copy(),
            vertices,
            faces,
            R_tokenhpe,
            t_gt,
            R_cam,
            t_cam,
            K_rgb,
            color="#d98533",
            alpha=0.5,
        )
        panels.append(panel_tokenhpe)
        titles.append(_title("TokenHPE", euler_tokenhpe))

    if R_whenet is not None:
        panel_whenet = render_mesh_on_image(
            img_rgb.copy(),
            vertices,
            faces,
            R_whenet,
            t_gt,
            R_cam,
            t_cam,
            K_rgb,
            color="#e2b714",
            alpha=0.5,
        )
        panels.append(panel_whenet)
        titles.append(_title("WHENet", euler_whenet))

    if R_trg is not None:
        panel_trg = render_mesh_on_image(
            img_rgb.copy(),
            vertices,
            faces,
            R_trg,
            t_gt,
            R_cam,
            t_cam,
            K_rgb,
            color="#357edd",
            alpha=0.5,
        )
        panels.append(panel_trg)
        titles.append(_title("TRG", euler_trg))

    n_cols = len(panels)
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))
    for ax, panel, title in zip(axes, panels, titles):
        ax.imshow(panel)
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════════
# 8.  MAIN EVALUATION LOOP
# ═════════════════════════════════════════════════════════════════════════════

SIXD_TRANSFORM = TF.Compose([
    TF.Resize(256), TF.CenterCrop(224), TF.ToTensor(),
    TF.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def vggt_transform(crop_np, target_size=518):
    pil = Image.fromarray(crop_np).resize((target_size, target_size), Image.BICUBIC)
    return TF.ToTensor()(pil)


def evaluate(args):
    device    = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    amp_dtype = (torch.bfloat16
                 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
                 else torch.float16)
    variant_names = get_variant_names(args.vggt_pose_mode)
    primary_variant = get_primary_variant_name(args.vggt_pose_mode)

    test_subjects = set(args.test_subjects)
    logging.info(f"Device: {device}  |  AMP: {amp_dtype}")
    logging.info(f"Test subjects: {sorted(test_subjects)}")
    logging.info(f"VGGT pose mode: {args.vggt_pose_mode}")
    os.makedirs(args.vis_dir, exist_ok=True)

    # ── Load models ───────────────────────────────────────────────────────────
    lora_path  = None if args.no_lora else args.lora_checkpoint
    vggt_model = load_vggt_model(args.base_checkpoint, lora_path, device,
                                 enable_track=args.enable_track)

    sixd_model = None
    mtcnn_detector = None
    if not args.no_sixdrepnet:
        sixd_model = load_sixdrepnet_model(
            args.sixdrepnet_checkpoint, args.sixdrepnet_dir, device)
        # Initialise MTCNN for 6DRepNet face crops (pure PyTorch, no TF)
        from facenet_pytorch import MTCNN as FacenetMTCNN
        mtcnn_detector = FacenetMTCNN(keep_all=True, device=device)
        logging.info("MTCNN detector initialised for 6DRepNet crops (facenet-pytorch)")

    # ── Mesh / calibration cache ──────────────────────────────────────────────
    mesh_cache, cal_cache = {}, {}

    def get_mesh(subj):
        if subj not in mesh_cache:
            mesh_cache[subj] = load_obj(os.path.join(args.biwi_dir, f"{subj}.obj"))
        return mesh_cache[subj]

    def get_cal(subj):
        if subj not in cal_cache:
            cal_cache[subj] = load_rgb_intrinsics(
                os.path.join(args.biwi_dir, subj, "rgb.cal"))
        return cal_cache[subj]

    # ── Discover test subject directories ─────────────────────────────────────
    all_dirs  = sorted(glob.glob(os.path.join(args.biwi_dir, "*")))
    test_dirs = [d for d in all_dirs
                 if os.path.isdir(d) and os.path.basename(d).isdigit()
                 and int(os.path.basename(d)) in test_subjects]
    logging.info(f"Evaluating {len(test_dirs)} test subjects\n")

    # ── Global accumulators ───────────────────────────────────────────────────
    vggt_tot_y = vggt_tot_p = vggt_tot_r = 0.0
    sixd_tot_y = sixd_tot_p = sixd_tot_r = 0.0
    total_count = 0
    vis_count   = 0

    per_subject = {}
    global_variant_errors = _empty_variant_acc(variant_names)

    # ── Per-subject loop ──────────────────────────────────────────────────────
    for subj_dir in test_dirs:
        subj = os.path.basename(subj_dir)

        if not os.path.exists(os.path.join(args.biwi_dir, f"{subj}.obj")):
            logging.warning(f"Subject {subj}: no .obj mesh, skipping.")
            continue
        if not os.path.exists(os.path.join(subj_dir, "rgb.cal")):
            logging.warning(f"Subject {subj}: no rgb.cal, skipping.")
            continue

        frames = get_subject_frames(subj_dir)
        use_anchor_pair = args.vggt_pose_mode in {"relative_h2c", "absolute_second"}
        min_required_frames = 2 if use_anchor_pair else 1
        if len(frames) < min_required_frames:
            logging.warning(f"Subject {subj}: <{min_required_frames} frames, skipping.")
            continue

        vertices, faces     = get_mesh(subj)
        K_rgb, R_cam, t_cam = get_cal(subj)

        anchor_tensor = None
        R_anchor_t = None
        if use_anchor_pair:
            _, anchor_rgb_path, anchor_pose_path = frames[0]
            R_anchor, t_anchor = parse_pose_txt(anchor_pose_path)
            anchor_img    = load_img_rgb(anchor_rgb_path)
            anchor_crop   = crop_face(anchor_img, t_anchor, K_rgb, args.crop_size)
            anchor_tensor = vggt_transform(anchor_crop)
            R_anchor_t    = torch.from_numpy(R_anchor).float()
        R_cam_t       = torch.from_numpy(R_cam).float()

        target_frames = frames[1:] if use_anchor_pair else frames
        n_targets     = len(target_frames)
        n_batches     = (n_targets + args.batch_size - 1) // args.batch_size

        if use_anchor_pair:
            logging.info(f"[{subj}] anchor=frame {frames[0][0]}, "
                         f"{n_targets} targets, {n_batches} batches")
        else:
            logging.info(f"[{subj}] single-view evaluation, "
                         f"{n_targets} frames, {n_batches} batches")

        # Per-subject accumulators
        s_variant_errors = _empty_variant_acc(variant_names)
        s_sixd_y = s_sixd_p = s_sixd_r = 0.0
        s_count  = 0

        # MTCNN previous-box tracker (reset per subject)
        mtcnn_prev_box = None

        for b_idx, b_start in enumerate(range(0, n_targets, args.batch_size)):
            batch = target_frames[b_start: b_start + args.batch_size]
            bs    = len(batch)

            if b_idx % 20 == 0:
                logging.info(f"  batch {b_idx+1}/{n_batches}")

            # ── Load batch ────────────────────────────────────────────────────
            vggt_tensors, sixd_tensors = [], []
            R_targets, t_targets       = [], []
            imgs_rgb                   = []

            for _, rgb_path, pose_path in batch:
                R_tgt, t_tgt = parse_pose_txt(pose_path)
                img_rgb      = load_img_rgb(rgb_path)

                # VGGT crop: 3D-projection based
                vggt_crop = crop_face(img_rgb, t_tgt, K_rgb, args.crop_size)
                vggt_tensors.append(vggt_transform(vggt_crop))

                # 6DRepNet crop: MTCNN-based (matching npz training data)
                if sixd_model is not None:
                    img_bgr = cv2.imread(rgb_path)
                    mtcnn_crop_bgr, mtcnn_prev_box = crop_face_mtcnn(
                        img_bgr, mtcnn_detector, mtcnn_prev_box,
                        crop_size=args.crop_size, ad=args.mtcnn_ad)
                    sixd_tensors.append(
                        SIXD_TRANSFORM(Image.fromarray(mtcnn_crop_bgr)))

                R_targets.append(torch.from_numpy(R_tgt).float())
                t_targets.append(t_tgt)
                imgs_rgb.append(img_rgb)

            R_gt_batch = torch.stack(R_targets)
            y_gt, p_gt, r_gt = biwi_euler_deg_batch(R_gt_batch)

            # ── VGGT inference ────────────────────────────────────────────────
            tgt_stack   = torch.stack(vggt_tensors)
            if use_anchor_pair:
                anchor_rep  = anchor_tensor.unsqueeze(0).expand(bs, -1, -1, -1)
                vggt_images = torch.stack([anchor_rep, tgt_stack], dim=1).to(device)
            else:
                vggt_images = tgt_stack.unsqueeze(1).to(device)

            with torch.no_grad(), torch.cuda.amp.autocast(dtype=amp_dtype):
                vggt_out = vggt_model(images=vggt_images)

            variants = decode_vggt_pose(
                vggt_out["pose_enc"],
                R_anchor_t,
                R_cam_t,
                pose_mode=args.vggt_pose_mode,
            )

            for vname, R_v in variants.items():
                yv, pv, rv = biwi_euler_deg_batch(R_v)
                err_y = angular_error(y_gt, yv).sum().item()
                err_p = angular_error(p_gt, pv).sum().item()
                err_r = angular_error(r_gt, rv).sum().item()
                global_variant_errors[vname]["y"] += err_y
                global_variant_errors[vname]["p"] += err_p
                global_variant_errors[vname]["r"] += err_r
                s_variant_errors[vname]["y"]      += err_y
                s_variant_errors[vname]["p"]      += err_p
                s_variant_errors[vname]["r"]      += err_r

            R_vggt_primary = variants[primary_variant]
            y_vp, p_vp, r_vp = biwi_euler_deg_batch(R_vggt_primary)
            vggt_tot_y += angular_error(y_gt, y_vp).sum().item()
            vggt_tot_p += angular_error(p_gt, p_vp).sum().item()
            vggt_tot_r += angular_error(r_gt, r_vp).sum().item()

            # ── 6DRepNet inference ────────────────────────────────────────────
            y_sp = p_sp = r_sp = None
            R_sixd_biwi = None
            if sixd_model is not None:
                sixd_imgs = torch.stack(sixd_tensors).to(device)
                with torch.no_grad():
                    R_sixd_batch = sixd_model(sixd_imgs).cpu()
                y_sp, p_sp, r_sp, R_sixd_biwi = decode_sixdrepnet_pose(R_sixd_batch)

                s_sixd_y += angular_error(y_gt, y_sp).sum().item()
                s_sixd_p += angular_error(p_gt, p_sp).sum().item()
                s_sixd_r += angular_error(r_gt, r_sp).sum().item()
                sixd_tot_y += angular_error(y_gt, y_sp).sum().item()
                sixd_tot_p += angular_error(p_gt, p_sp).sum().item()
                sixd_tot_r += angular_error(r_gt, r_sp).sum().item()

            s_count += bs

            # ── Visualization ─────────────────────────────────────────────────
            if args.visualize:
                for fi in range(bs):
                    if vis_count % args.vis_every_n == 0:
                        frame_num = batch[fi][0]
                        save_path = os.path.join(
                            args.vis_dir, f"subj{subj}_frame{frame_num:05d}.png")

                        euler_gt_i = (y_gt[fi].item(), p_gt[fi].item(), r_gt[fi].item())

                        vggt_vis = {}
                        for vname, R_v in variants.items():
                            yv, pv, rv = biwi_euler_deg_batch(R_v)
                            vggt_vis[vname] = (
                                R_v[fi].numpy(),
                                (yv[fi].item(), pv[fi].item(), rv[fi].item()),
                            )

                        euler_sixd_i = (y_sp[fi].item(), p_sp[fi].item(), r_sp[fi].item()) \
                                        if y_sp is not None else None

                        save_vis_frame(
                            img_rgb       = imgs_rgb[fi],
                            vertices      = vertices,
                            faces         = faces,
                            K_rgb         = K_rgb,
                            R_cam         = R_cam,
                            t_cam         = t_cam,
                            R_gt          = R_gt_batch[fi].numpy(),
                            t_gt          = t_targets[fi],
                            euler_gt      = euler_gt_i,
                            vggt_variants = vggt_vis,
                            R_sixd        = R_sixd_biwi[fi] if R_sixd_biwi is not None else None,
                            euler_sixd    = euler_sixd_i,
                            save_path     = save_path,
                        )
                    vis_count += 1

        # ── Per-subject summary ───────────────────────────────────────────────
        if s_count > 0:
            per_subject[subj] = {}

            # All 4 VGGT variants
            for vname in variant_names:
                vy = s_variant_errors[vname]["y"] / s_count
                vp = s_variant_errors[vname]["p"] / s_count
                vr = s_variant_errors[vname]["r"] / s_count
                vm = (vy + vp + vr) / 3
                per_subject[subj][f"vggt_{vname}"] = [vy, vp, vr, vm]

            vy, vp, vr, vm = per_subject[subj][f"vggt_{primary_variant}"]
            line = (f"Subject {subj} ({s_count} frames):\n"
                    f"  VGGT ({primary_variant}) — Yaw: {vy:.2f}  Pitch: {vp:.2f}  Roll: {vr:.2f}  MAE: {vm:.2f}")

            if "vggt_B_no_flip" in per_subject[subj]:
                by, bp, br, bm = per_subject[subj]["vggt_B_no_flip"]
                line += (f"\n  VGGT (B_no_flip) — Yaw: {by:.2f}  Pitch: {bp:.2f}"
                         f"  Roll: {br:.2f}  MAE: {bm:.2f}")

            if sixd_model is not None:
                sy = s_sixd_y / s_count
                sp = s_sixd_p / s_count
                sr = s_sixd_r / s_count
                sm = (sy + sp + sr) / 3
                per_subject[subj]["sixd"] = [sy, sp, sr, sm]
                line += (f"\n  6DRepNet  — Yaw: {sy:.2f}  Pitch: {sp:.2f}"
                         f"  Roll: {sr:.2f}  MAE: {sm:.2f}")
            logging.info(line)

        total_count += s_count
        gc.collect(); torch.cuda.empty_cache()

    # ── Overall results ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("BIWI EVALUATION RESULTS")
    print("=" * 60)
    if total_count > 0:
        vy = vggt_tot_y / total_count
        vp = vggt_tot_p / total_count
        vr = vggt_tot_r / total_count
        vm = (vy + vp + vr) / 3
        print(f"Frames evaluated: {total_count}")
        print(f"\nVGGT ({primary_variant}) — Yaw: {vy:.4f}  Pitch: {vp:.4f}  Roll: {vr:.4f}  MAE: {vm:.4f}")
        if sixd_model is not None:
            sy = sixd_tot_y / total_count
            sp = sixd_tot_p / total_count
            sr = sixd_tot_r / total_count
            sm = (sy + sp + sr) / 3
            print(f"6DRepNet           — Yaw: {sy:.4f}  Pitch: {sp:.4f}  Roll: {sr:.4f}  MAE: {sm:.4f}")

        print(f"\n{'─'*60}")
        print("VGGT variant comparison:")
        print(f"{'─'*60}")
        best_name, best_mae = None, float("inf")
        for vname in variant_names:
            errs = global_variant_errors[vname]
            vy_ = errs["y"] / total_count
            vp_ = errs["p"] / total_count
            vr_ = errs["r"] / total_count
            vm_ = (vy_ + vp_ + vr_) / 3
            if vm_ < best_mae:
                best_mae  = vm_
                best_name = vname
            marker = " ← best" if vname == best_name else ""
            print(f"  {vname:15s} — Yaw: {vy_:.4f}  Pitch: {vp_:.4f}  Roll: {vr_:.4f}  MAE: {vm_:.4f}{marker}")
        print(f"\n  ✓ Best variant: {best_name}  (MAE: {best_mae:.4f}°)")
    else:
        print("No frames evaluated.")
    print("=" * 60)

    # ── Save CSV ──────────────────────────────────────────────────────────────
    csv_path = os.path.join(args.vis_dir, "biwi_eval_results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)

        # ── Per-subject rows ──────────────────────────────────────────────────
        w.writerow(["subject", "model", "yaw_mae", "pitch_mae", "roll_mae", "mean_mae"])
        for subj, res in per_subject.items():
            for model_name, vals in res.items():
                w.writerow([subj, model_name] + [f"{v:.4f}" for v in vals])

        if total_count > 0:
            # ── Overall rows ──────────────────────────────────────────────────
            w.writerow([])
            w.writerow(["# Overall results"])
            w.writerow(["subject", "model", "yaw_mae", "pitch_mae", "roll_mae", "mean_mae"])
            w.writerow(["overall", f"vggt_{primary_variant}",
                        f"{vggt_tot_y/total_count:.4f}",
                        f"{vggt_tot_p/total_count:.4f}",
                        f"{vggt_tot_r/total_count:.4f}",
                        f"{(vggt_tot_y+vggt_tot_p+vggt_tot_r)/(total_count*3):.4f}"])
            if sixd_model is not None:
                w.writerow(["overall", "sixdrepnet",
                            f"{sixd_tot_y/total_count:.4f}",
                            f"{sixd_tot_p/total_count:.4f}",
                            f"{sixd_tot_r/total_count:.4f}",
                            f"{(sixd_tot_y+sixd_tot_p+sixd_tot_r)/(total_count*3):.4f}"])

            # ── Global variant comparison ─────────────────────────────────────
            w.writerow([])
            w.writerow(["# VGGT variants (global)"])
            w.writerow(["variant", "model", "yaw_mae", "pitch_mae", "roll_mae", "mean_mae"])
            best_name, best_mae = None, float("inf")
            for vname in variant_names:
                errs = global_variant_errors[vname]
                vy_ = errs["y"] / total_count
                vp_ = errs["p"] / total_count
                vr_ = errs["r"] / total_count
                vm_ = (vy_ + vp_ + vr_) / 3
                w.writerow([vname, "vggt",
                            f"{vy_:.4f}", f"{vp_:.4f}", f"{vr_:.4f}", f"{vm_:.4f}"])
                if vm_ < best_mae:
                    best_mae  = vm_
                    best_name = vname
            w.writerow(["best_variant", best_name, "", "", "", f"{best_mae:.4f}"])

    logging.info(f"Results saved → {csv_path}")


# ═════════════════════════════════════════════════════════════════════════════
# 9.  CLI
# ═════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="BIWI evaluation: VGGT + 6DRepNet")

    p.add_argument("--biwi_dir",      default="/gpu-data4/filby/head_pose_datasets/faces_0")
    p.add_argument("--test_subjects", type=int, nargs="+", default=DEFAULT_TEST_SUBJECTS)
    p.add_argument("--crop_size",     type=int, default=256)

    p.add_argument("--base_checkpoint",
                   default=os.path.join(SCRIPT_DIR, "checkpoints", "VGGT-1B", "model.pt"))
    p.add_argument("--lora_checkpoint",
                   default=os.path.join(SCRIPT_DIR, "training", "logs",
                                        "flame_h2c_lora", "ckpts", "checkpoint.pt"))
    p.add_argument("--no_lora",       action="store_true")
    p.add_argument(
        "--vggt_pose_mode",
        choices=["relative_h2c", "absolute_second", "absolute_single"],
        default="relative_h2c",
        help="How to decode the target-frame VGGT prediction during BIWI evaluation.",
    )
    p.add_argument("--enable_track",  action="store_true",
                   help="Load pretrained track head (no query_points → head is dormant at eval)")

    p.add_argument("--sixdrepnet_dir",
                   default=os.path.join(SCRIPT_DIR, "..", "6DRepNet", "sixdrepnet"))
    p.add_argument("--sixdrepnet_checkpoint",
                   default=os.path.join(SCRIPT_DIR, "..", "6DRepNet",
                                        "sixdrepnet", "6DRepNet_70_30_BIWI.pth"))
    p.add_argument("--no_sixdrepnet", action="store_true")

    p.add_argument("--mtcnn_ad",    type=float, default=0.4,
                   help="MTCNN crop enlargement margin (must match npz creation)")

    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--gpu",        type=int, default=0)

    p.add_argument("--visualize",   action="store_true")
    p.add_argument("--vis_dir",     default=os.path.join(SCRIPT_DIR, "vis_output"))
    p.add_argument("--vis_every_n", type=int, default=1)

    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
