# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
FLAME Head-to-Camera (H2C) Dataset for VGGT training.

This dataset loads rendered FLAME head images and computes head-to-camera (H2C)
transformations as the extrinsics. This allows training VGGT to predict relative
H2C poses between two views, where both the head and camera can move.

The H2C transformation captures the combined effect of:
1. Head pose in world space (can change due to expression/pose)
2. Camera pose in world space (can change due to camera movement)

Data hierarchy expected:
    root_dir/
        identity_XXXXXX/
            lighting_XXX/
                expr_XXX/
                    view_XXX/
                        output.png
                        output_opencv_camera.pkl           (K, R, t in OpenCV convention)
                        output_object_transform_post_render.pkl  (R_world, t_world of head)
"""

import os
import os.path as osp
import logging
import random
import glob
import pickle
from pathlib import Path

import cv2
import numpy as np

from data.base_dataset import BaseDataset
from data.dataset_util import read_image_cv2


class FlameH2CDataset(BaseDataset):
    """
    Dataset for FLAME head images with Head-to-Camera (H2C) pose supervision.

    Instead of providing world-to-camera extrinsics, this dataset provides
    head-to-camera (H2C) transformations. This allows the model to learn
    relative H2C poses between views.
    """

    def __init__(
        self,
        common_conf,
        split: str = "train",
        root_dir: str = "/gpu-data3/vvas/blended_in_flames/sample_outputs/ar_pose_poc_5k",
        len_train: int = 100000,
        len_test: int = 10000,
        num_views: int = 2,  # Number of views per sample (typically 2 for relative pose)
        same_identity: bool = True,  # Sample views from same identity
        same_lighting: bool = True,  # Sample views from same lighting condition
        same_expression: bool = False,  # If True, only camera changes; if False, expression can also change
        depth_max: float = 10.0,  # Max depth value (heads are close, so smaller than VKitti)
    ):
        """
        Initialize the FlameH2CDataset.

        Args:
            common_conf: Configuration object with common settings.
            split (str): Dataset split, either 'train' or 'test'.
            root_dir (str): Root directory containing FLAME rendered data.
            len_train (int): Length of the training dataset.
            len_test (int): Length of the test dataset.
            num_views (int): Number of views to sample per sequence.
            same_identity (bool): Whether to sample views from the same identity.
            same_lighting (bool): Whether to sample views from the same lighting.
            same_expression (bool): Whether to sample views from the same expression.
            depth_max (float): Maximum depth value for depth maps.
        """
        super().__init__(common_conf=common_conf)

        self.debug = common_conf.debug
        self.training = common_conf.training
        self.root_dir = root_dir
        self.num_views = num_views
        self.same_identity = same_identity
        self.same_lighting = same_lighting
        self.same_expression = same_expression
        self.depth_max = depth_max

        if split == "train":
            self.len_train = len_train
        elif split == "test":
            self.len_train = len_test
        else:
            raise ValueError(f"Invalid split: {split}")

        logging.info(f"FlameH2C root_dir is {self.root_dir}")

        # Build index of all views
        self._build_index()

        status = "Training" if self.training else "Testing"
        logging.info(f"{status}: FlameH2C dataset size: {len(self.all_views)}")
        logging.info(f"{status}: FlameH2C dataset length: {len(self)}")

    def _build_index(self):
        """Build hierarchical index of all views."""
        self.identities = {}
        self.all_views = []

        identity_dirs = sorted(glob.glob(osp.join(self.root_dir, "identity_*")))

        for identity_dir in identity_dirs:
            identity_name = osp.basename(identity_dir)
            self.identities[identity_name] = {"lightings": {}}

            lighting_dirs = sorted(glob.glob(osp.join(identity_dir, "lighting_*")))
            for lighting_dir in lighting_dirs:
                lighting_name = osp.basename(lighting_dir)
                self.identities[identity_name]["lightings"][lighting_name] = {"expressions": {}}

                expr_dirs = sorted(glob.glob(osp.join(lighting_dir, "expr_*")))
                for expr_dir in expr_dirs:
                    expr_name = osp.basename(expr_dir)
                    self.identities[identity_name]["lightings"][lighting_name]["expressions"][expr_name] = {"views": []}

                    view_dirs = sorted(glob.glob(osp.join(expr_dir, "view_*")))
                    for view_dir in view_dirs:
                        view_name = osp.basename(view_dir)
                        view_info = {
                            "path": view_dir,
                            "identity": identity_name,
                            "lighting": lighting_name,
                            "expression": expr_name,
                            "view": view_name,
                        }
                        self.identities[identity_name]["lightings"][lighting_name]["expressions"][expr_name]["views"].append(view_info)
                        self.all_views.append(view_info)

        self.identity_list = list(self.identities.keys())

    def _load_h2c(self, view_dir: str) -> tuple:
        """
        Load head-to-camera (H2C) transformation from view directory.

        Uses the pre-converted OpenCV camera pkl (output_opencv_camera.pkl) and
        the object transform pkl. Both are in Blender world coordinates, so they
        compose directly without any coordinate system conversion.

        Args:
            view_dir: Path to view directory containing pkl files.

        Returns:
            R_h2c: 3x3 rotation matrix (head to camera, OpenCV convention)
            t_h2c: 3D translation vector (head to camera, OpenCV convention)
        """
        cam_path = osp.join(view_dir, "output_opencv_camera.pkl")
        obj_path = osp.join(view_dir, "output_object_transform_post_render.pkl")

        with open(cam_path, "rb") as f:
            cam = pickle.load(f)
        with open(obj_path, "rb") as f:
            obj = pickle.load(f)

        # Camera: world-to-camera in OpenCV convention (already converted)
        R_w2c = np.array(cam["R"], dtype=np.float64)
        t_w2c = np.array(cam["t"], dtype=np.float64).flatten()

        # Object (head): object-to-world in Blender world coords (same space as camera)
        R_h2w = np.array(obj["R_world"], dtype=np.float64)
        t_h2w = np.array(obj["t_world"], dtype=np.float64).flatten()

        # Compose: head -> world -> camera
        R_h2c = R_w2c @ R_h2w
        t_h2c = R_w2c @ t_h2w + t_w2c

        return R_h2c, t_h2c

    def _load_intrinsics(self, view_dir: str) -> np.ndarray:
        """Load camera intrinsics from view directory."""
        cam_path = osp.join(view_dir, "output_opencv_camera.pkl")
        with open(cam_path, "rb") as f:
            cam = pickle.load(f)

        K = np.array(cam["K"], dtype=np.float64).reshape(3, 3)
        return K

    def _sample_view_pair(self) -> list:
        """
        Sample a pair of views based on configuration.

        Returns:
            List of view_info dicts for each sampled view.
        """
        views = []

        # Sample first view randomly
        view1 = random.choice(self.all_views)
        views.append(view1)

        # Sample second view based on constraints
        candidates = self.all_views

        if self.same_identity:
            candidates = [v for v in candidates if v["identity"] == view1["identity"]]

        if self.same_lighting:
            candidates = [v for v in candidates if v["lighting"] == view1["lighting"]]

        if self.same_expression:
            candidates = [v for v in candidates if v["expression"] == view1["expression"]]

        # Remove view1 from candidates to ensure different views
        candidates = [v for v in candidates if v["path"] != view1["path"]]

        if len(candidates) == 0:
            # Fallback: just use a different view from same expression
            expr_views = self.identities[view1["identity"]]["lightings"][view1["lighting"]]["expressions"][view1["expression"]]["views"]
            candidates = [v for v in expr_views if v["path"] != view1["path"]]

            if len(candidates) == 0:
                # Last resort: duplicate view1 (not ideal but prevents crash)
                candidates = [view1]

        view2 = random.choice(candidates)
        views.append(view2)

        return views

    def get_data(
        self,
        seq_index: int = None,
        img_per_seq: int = None,
        seq_name: str = None,
        ids: list = None,
        aspect_ratio: float = 1.0,
    ) -> dict:
        """
        Retrieve data for a view pair.

        Args:
            seq_index (int): Index of the sequence (used for deterministic sampling).
            img_per_seq (int): Number of images per sequence (ignored, uses self.num_views).
            seq_name (str): Name of the sequence (not used for this dataset).
            ids (list): Specific view indices (not used for this dataset).
            aspect_ratio (float): Aspect ratio for image processing.

        Returns:
            dict: A batch of data including images, H2C extrinsics, intrinsics, etc.
        """
        # Sample view pair
        view_infos = self._sample_view_pair()

        target_image_shape = self.get_target_shape(aspect_ratio)

        images = []
        depths = []
        cam_points = []
        world_points = []
        point_masks = []
        extrinsics = []  # H2C matrices
        intrinsics = []
        original_sizes = []

        for view_info in view_infos:
            view_dir = view_info["path"]

            # Load image
            image_path = osp.join(view_dir, "output.png")
            image = read_image_cv2(image_path)
            original_size = np.array(image.shape[:2])

            # Load H2C as extrinsics (this is the key difference!)
            R_h2c, t_h2c = self._load_h2c(view_dir)
            extri_h2c = np.zeros((3, 4), dtype=np.float64)
            extri_h2c[:3, :3] = R_h2c
            extri_h2c[:3, 3] = t_h2c

            # Load intrinsics
            intri = self._load_intrinsics(view_dir)

            # Create dummy depth map (we don't have depth GT for FLAME renders)
            # Set to ones - the normalization will handle scaling
            depth_map = np.ones(original_size, dtype=np.float32)

            # Process image (resize, crop, etc.)
            (
                image,
                depth_map,
                extri_h2c,
                intri,
                world_coords_points,
                cam_coords_points,
                point_mask,
                _,
            ) = self.process_one_image(
                image,
                depth_map,
                extri_h2c,
                intri,
                original_size,
                target_image_shape,
                filepath=image_path,
            )

            if (image.shape[:2] != target_image_shape).any():
                logging.warning(f"Wrong shape for {view_dir}: expected {target_image_shape}, got {image.shape[:2]}")
                continue

            images.append(image)
            depths.append(depth_map)
            extrinsics.append(extri_h2c)
            intrinsics.append(intri)
            cam_points.append(cam_coords_points)
            world_points.append(world_coords_points)
            point_masks.append(point_mask)
            original_sizes.append(original_size)

        # Build sequence name from view paths
        seq_name = f"flame_h2c_{view_infos[0]['identity']}_{view_infos[0]['lighting']}"

        # ids must be a numpy array of integers for the composed_dataset
        ids_array = np.arange(len(view_infos))

        batch = {
            "seq_name": seq_name,
            "ids": ids_array,
            "frame_num": len(extrinsics),
            "images": images,
            "depths": depths,
            "extrinsics": extrinsics,  # H2C matrices!
            "intrinsics": intrinsics,
            "cam_points": cam_points,
            "world_points": world_points,
            "point_masks": point_masks,
            "original_sizes": original_sizes,
            # Store view paths for debugging
            "view_paths": [v["path"] for v in view_infos],
            # Mesh paths for visualization
            "mesh_paths": [osp.join(v["path"], "output_mesh.obj") for v in view_infos],
        }
        return batch
