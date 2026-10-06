import argparse
import csv
import gc
import glob
import logging
import os

import cv2
import torch
from PIL import Image

import eval_biwi_crop as base


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def crop_face_mtcnn_for_vggt(img_bgr, detector, prev_box, crop_size=256, ad=0.4):
    """
    Reuse the same MTCNN crop logic as 6DRepNet, but convert the cropped image back to RGB
    before feeding it to VGGT.
    """
    crop_bgr, new_box = base.crop_face_mtcnn(
        img_bgr,
        detector,
        prev_box,
        crop_size=crop_size,
        ad=ad,
    )
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    return crop_rgb, crop_bgr, new_box


def evaluate(args):
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
    logging.info("VGGT crop mode: shared MTCNN crop")
    os.makedirs(args.vis_dir, exist_ok=True)

    # Load models.
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

    # Always initialize MTCNN in this evaluator because VGGT also depends on it.
    from facenet_pytorch import MTCNN as FacenetMTCNN

    mtcnn_detector = FacenetMTCNN(keep_all=True, device=device)
    logging.info("MTCNN detector initialized for shared VGGT/6DRepNet crops")

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
    logging.info(f"Evaluating {len(test_dirs)} test subjects\n")

    vggt_tot_y = vggt_tot_p = vggt_tot_r = 0.0
    sixd_tot_y = sixd_tot_p = sixd_tot_r = 0.0
    total_count = 0
    vis_count = 0

    per_subject = {}
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

        vertices, faces = get_mesh(subj)
        K_rgb, R_cam, t_cam = get_cal(subj)

        anchor_tensor = None
        R_anchor_t = None
        if use_anchor_pair:
            _, anchor_rgb_path, anchor_pose_path = frames[0]
            R_anchor, _ = base.parse_pose_txt(anchor_pose_path)
            anchor_bgr = cv2.imread(anchor_rgb_path)
            if anchor_bgr is None:
                logging.warning(f"Subject {subj}: failed to read anchor frame {anchor_rgb_path}, skipping.")
                continue
            anchor_crop_rgb, _, mtcnn_prev_box = crop_face_mtcnn_for_vggt(
                anchor_bgr,
                mtcnn_detector,
                None,
                crop_size=args.crop_size,
                ad=args.mtcnn_ad,
            )
            anchor_tensor = base.vggt_transform(anchor_crop_rgb)
            R_anchor_t = torch.from_numpy(R_anchor).float()
        R_cam_t = torch.from_numpy(R_cam).float()
        # Match the baseline 6DRepNet behavior: target-frame tracking starts from no previous box.
        mtcnn_prev_box = None

        target_frames = frames[1:] if use_anchor_pair else frames
        n_targets = len(target_frames)
        n_batches = (n_targets + args.batch_size - 1) // args.batch_size

        if use_anchor_pair:
            logging.info(f"[{subj}] anchor=frame {frames[0][0]}, {n_targets} targets, {n_batches} batches")
        else:
            logging.info(f"[{subj}] single-view evaluation, {n_targets} frames, {n_batches} batches")

        s_variant_errors = base._empty_variant_acc(variant_names)
        s_sixd_y = s_sixd_p = s_sixd_r = 0.0
        s_count = 0

        for b_idx, b_start in enumerate(range(0, n_targets, args.batch_size)):
            batch = target_frames[b_start : b_start + args.batch_size]
            bs = len(batch)

            if b_idx % 20 == 0:
                logging.info(f"  batch {b_idx + 1}/{n_batches}")

            vggt_tensors, sixd_tensors = [], []
            frame_nums, R_targets, t_targets = [], [], []
            imgs_rgb = []

            for frame_num, rgb_path, pose_path in batch:
                R_tgt, t_tgt = base.parse_pose_txt(pose_path)
                img_rgb = base.load_img_rgb(rgb_path)
                img_bgr = cv2.imread(rgb_path)
                if img_bgr is None:
                    logging.warning(f"Failed to read target frame {rgb_path}, skipping frame.")
                    continue

                shared_crop_rgb, shared_crop_bgr, mtcnn_prev_box = crop_face_mtcnn_for_vggt(
                    img_bgr,
                    mtcnn_detector,
                    mtcnn_prev_box,
                    crop_size=args.crop_size,
                    ad=args.mtcnn_ad,
                )
                vggt_tensors.append(base.vggt_transform(shared_crop_rgb))

                if sixd_model is not None:
                    sixd_tensors.append(base.SIXD_TRANSFORM(Image.fromarray(shared_crop_bgr)))

                frame_nums.append(frame_num)
                R_targets.append(torch.from_numpy(R_tgt).float())
                t_targets.append(t_tgt)
                imgs_rgb.append(img_rgb)

            if len(vggt_tensors) == 0:
                continue

            bs = len(vggt_tensors)
            R_gt_batch = torch.stack(R_targets)
            y_gt, p_gt, r_gt = base.biwi_euler_deg_batch(R_gt_batch)

            tgt_stack = torch.stack(vggt_tensors)
            if use_anchor_pair:
                anchor_rep = anchor_tensor.unsqueeze(0).expand(bs, -1, -1, -1)
                vggt_images = torch.stack([anchor_rep, tgt_stack], dim=1).to(device)
            else:
                vggt_images = tgt_stack.unsqueeze(1).to(device)

            with torch.no_grad(), torch.cuda.amp.autocast(dtype=amp_dtype):
                vggt_out = vggt_model(images=vggt_images)

            variants = base.decode_vggt_pose(
                vggt_out["pose_enc"],
                R_anchor_t,
                R_cam_t,
                pose_mode=args.vggt_pose_mode,
            )

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
            vggt_tot_y += base.angular_error(y_gt, y_vp).sum().item()
            vggt_tot_p += base.angular_error(p_gt, p_vp).sum().item()
            vggt_tot_r += base.angular_error(r_gt, r_vp).sum().item()

            y_sp = p_sp = r_sp = None
            R_sixd_biwi = None
            if sixd_model is not None:
                sixd_imgs = torch.stack(sixd_tensors).to(device)
                with torch.no_grad():
                    R_sixd_batch = sixd_model(sixd_imgs).cpu()
                y_sp, p_sp, r_sp, R_sixd_biwi = base.decode_sixdrepnet_pose(R_sixd_batch)

                s_sixd_y += base.angular_error(y_gt, y_sp).sum().item()
                s_sixd_p += base.angular_error(p_gt, p_sp).sum().item()
                s_sixd_r += base.angular_error(r_gt, r_sp).sum().item()
                sixd_tot_y += base.angular_error(y_gt, y_sp).sum().item()
                sixd_tot_p += base.angular_error(p_gt, p_sp).sum().item()
                sixd_tot_r += base.angular_error(r_gt, r_sp).sum().item()

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

                        base.save_vis_frame(
                            img_rgb=imgs_rgb[fi],
                            vertices=vertices,
                            faces=faces,
                            K_rgb=K_rgb,
                            R_cam=R_cam,
                            t_cam=t_cam,
                            R_gt=R_gt_batch[fi].numpy(),
                            t_gt=t_targets[fi],
                            euler_gt=euler_gt_i,
                            vggt_variants=vggt_vis,
                            R_sixd=R_sixd_biwi[fi] if R_sixd_biwi is not None else None,
                            euler_sixd=euler_sixd_i,
                            save_path=save_path,
                        )
                    vis_count += 1

        if s_count > 0:
            per_subject[subj] = {}
            for vname in variant_names:
                vy = s_variant_errors[vname]["y"] / s_count
                vp = s_variant_errors[vname]["p"] / s_count
                vr = s_variant_errors[vname]["r"] / s_count
                vm = (vy + vp + vr) / 3
                per_subject[subj][f"vggt_{vname}"] = [vy, vp, vr, vm]

            vy, vp, vr, vm = per_subject[subj][f"vggt_{primary_variant}"]
            line = (
                f"Subject {subj} ({s_count} frames):\n"
                f"  VGGT ({primary_variant}) — Yaw: {vy:.2f}  Pitch: {vp:.2f}  Roll: {vr:.2f}  MAE: {vm:.2f}"
            )

            if "vggt_B_no_flip" in per_subject[subj]:
                by, bp, br, bm = per_subject[subj]["vggt_B_no_flip"]
                line += (
                    f"\n  VGGT (B_no_flip) — Yaw: {by:.2f}  Pitch: {bp:.2f}"
                    f"  Roll: {br:.2f}  MAE: {bm:.2f}"
                )

            if sixd_model is not None:
                sy = s_sixd_y / s_count
                sp = s_sixd_p / s_count
                sr = s_sixd_r / s_count
                sm = (sy + sp + sr) / 3
                per_subject[subj]["sixd"] = [sy, sp, sr, sm]
                line += f"\n  6DRepNet  — Yaw: {sy:.2f}  Pitch: {sp:.2f}  Roll: {sr:.2f}  MAE: {sm:.2f}"
            logging.info(line)

        total_count += s_count
        gc.collect()
        torch.cuda.empty_cache()

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
        print("No frames evaluated.")
    print("=" * 60)

    csv_path = os.path.join(args.vis_dir, "biwi_eval_results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subject", "model", "yaw_mae", "pitch_mae", "roll_mae", "mean_mae"])
        for subj, res in per_subject.items():
            for model_name, vals in res.items():
                w.writerow([subj, model_name] + [f"{v:.4f}" for v in vals])

        if total_count > 0:
            w.writerow([])
            w.writerow(["# Overall results"])
            w.writerow(["subject", "model", "yaw_mae", "pitch_mae", "roll_mae", "mean_mae"])
            w.writerow([
                "overall",
                f"vggt_{primary_variant}",
                f"{vggt_tot_y / total_count:.4f}",
                f"{vggt_tot_p / total_count:.4f}",
                f"{vggt_tot_r / total_count:.4f}",
                f"{(vggt_tot_y + vggt_tot_p + vggt_tot_r) / (total_count * 3):.4f}",
            ])
            if sixd_model is not None:
                w.writerow([
                    "overall",
                    "sixdrepnet",
                    f"{sixd_tot_y / total_count:.4f}",
                    f"{sixd_tot_p / total_count:.4f}",
                    f"{sixd_tot_r / total_count:.4f}",
                    f"{(sixd_tot_y + sixd_tot_p + sixd_tot_r) / (total_count * 3):.4f}",
                ])

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
                w.writerow([vname, "vggt", f"{vy_:.4f}", f"{vp_:.4f}", f"{vr_:.4f}", f"{vm_:.4f}"])
                if vm_ < best_mae:
                    best_mae = vm_
                    best_name = vname
            w.writerow(["best_variant", best_name, "", "", "", f"{best_mae:.4f}"])

    logging.info(f"Results saved -> {csv_path}")


def parse_args():
    p = argparse.ArgumentParser(description="BIWI evaluation with shared MTCNN crops for VGGT and 6DRepNet")

    p.add_argument("--biwi_dir", default="/gpu-data4/filby/head_pose_datasets/faces_0")
    p.add_argument("--test_subjects", type=int, nargs="+", default=base.DEFAULT_TEST_SUBJECTS)
    p.add_argument("--crop_size", type=int, default=256)

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
    p.add_argument("--enable_track", action="store_true", help="Load pretrained track head (no query_points -> dormant at eval)")

    p.add_argument("--sixdrepnet_dir", default=os.path.join(base.SCRIPT_DIR, "..", "6DRepNet", "sixdrepnet"))
    p.add_argument(
        "--sixdrepnet_checkpoint",
        default=os.path.join(base.SCRIPT_DIR, "..", "6DRepNet", "sixdrepnet", "6DRepNet_70_30_BIWI.pth"),
    )
    p.add_argument("--no_sixdrepnet", action="store_true")

    p.add_argument("--mtcnn_ad", type=float, default=0.4, help="Shared MTCNN crop enlargement margin")

    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--gpu", type=int, default=0)

    p.add_argument("--visualize", action="store_true")
    p.add_argument("--vis_dir", default=os.path.join(base.SCRIPT_DIR, "vis_output_mtcnn"))
    p.add_argument("--vis_every_n", type=int, default=1)

    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
