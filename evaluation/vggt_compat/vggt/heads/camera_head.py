# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from vggt.layers import Mlp
from vggt.layers.block import Block
from vggt.heads.head_act import activate_pose
from vggt.utils.pose_enc import get_pose_encoding_dim


class CameraHead(nn.Module):
    """
    CameraHead predicts camera parameters from token representations using iterative refinement.

    It applies a series of transformer blocks (the "trunk") to dedicated camera tokens.

    Supports two pose encoding types:
    - "absT_quaR_FoV": Translation (3) + Quaternion (4) + FoV (2) = 9D
    - "absT_eulerR_FoV": Translation (3) + Euler angles (3) + FoV (2) = 8D
    """

    def __init__(
        self,
        dim_in: int = 2048,
        trunk_depth: int = 4,
        pose_encoding_type: str = "absT_quaR_FoV",
        num_heads: int = 16,
        mlp_ratio: int = 4,
        init_values: float = 0.01,
        trans_act: str = "linear",
        quat_act: str = "linear",  # Also used for Euler angles
        fl_act: str = "relu",  # Field of view activations: ensures FOV values are positive.
        split_pose_branch: bool = False,
    ):
        super().__init__()

        self.pose_encoding_type = pose_encoding_type
        self.target_dim = get_pose_encoding_dim(pose_encoding_type)

        self.trans_act = trans_act
        self.quat_act = quat_act
        self.fl_act = fl_act
        self.trunk_depth = trunk_depth
        self.split_pose_branch = split_pose_branch
        self.trans_dim, self.rot_dim, self.fov_dim = _pose_component_dims(pose_encoding_type)

        # Build the trunk using a sequence of transformer blocks.
        self.trunk = nn.Sequential(
            *[
                Block(dim=dim_in, num_heads=num_heads, mlp_ratio=mlp_ratio, init_values=init_values)
                for _ in range(trunk_depth)
            ]
        )

        # Normalizations for camera token and trunk output.
        self.token_norm = nn.LayerNorm(dim_in)
        self.trunk_norm = nn.LayerNorm(dim_in)

        # Learnable empty camera pose token.
        self.empty_pose_tokens = nn.Parameter(torch.zeros(1, 1, self.target_dim))
        self.embed_pose = nn.Linear(self.target_dim, dim_in)

        # Module for producing modulation parameters: shift, scale, and a gate.
        self.poseLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim_in, 3 * dim_in, bias=True))

        # Adaptive layer normalization without affine parameters.
        self.adaln_norm = nn.LayerNorm(dim_in, elementwise_affine=False, eps=1e-6)
        if self.split_pose_branch:
            trans_fov_dim = self.trans_dim + self.fov_dim
            if trans_fov_dim == 0:
                raise ValueError(f"split_pose_branch requires translation or FoV dims for {pose_encoding_type}")
            self.pose_branch_trans_fov = Mlp(
                in_features=dim_in, hidden_features=dim_in // 2, out_features=trans_fov_dim, drop=0
            )
            self.pose_branch_rot = Mlp(
                in_features=dim_in, hidden_features=dim_in // 2, out_features=self.rot_dim, drop=0
            )
        else:
            self.pose_branch = Mlp(in_features=dim_in, hidden_features=dim_in // 2, out_features=self.target_dim, drop=0)

    def forward(self, aggregated_tokens_list: list, num_iterations: int = 4) -> list:
        """
        Forward pass to predict camera parameters.

        Args:
            aggregated_tokens_list (list): List of token tensors from the network;
                the last tensor is used for prediction.
            num_iterations (int, optional): Number of iterative refinement steps. Defaults to 4.

        Returns:
            list: A list of predicted camera encodings (post-activation) from each iteration.
        """
        # Use tokens from the last block for camera prediction.
        tokens = aggregated_tokens_list[-1]

        # Extract the camera tokens
        pose_tokens = tokens[:, :, 0]
        pose_tokens = self.token_norm(pose_tokens)

        pred_pose_enc_list = self.trunk_fn(pose_tokens, num_iterations)
        return pred_pose_enc_list

    def trunk_fn(self, pose_tokens: torch.Tensor, num_iterations: int) -> list:
        """
        Iteratively refine camera pose predictions.

        Args:
            pose_tokens (torch.Tensor): Normalized camera tokens with shape [B, S, C].
            num_iterations (int): Number of refinement iterations.

        Returns:
            list: List of activated camera encodings from each iteration.
        """
        B, S, C = pose_tokens.shape
        pred_pose_enc = None
        pred_pose_enc_list = []

        for _ in range(num_iterations):
            # Use a learned empty pose for the first iteration.
            if pred_pose_enc is None:
                module_input = self.embed_pose(self.empty_pose_tokens.expand(B, S, -1))
            else:
                # Detach the previous prediction to avoid backprop through time.
                pred_pose_enc = pred_pose_enc.detach()
                module_input = self.embed_pose(pred_pose_enc)

            # Generate modulation parameters and split them into shift, scale, and gate components.
            shift_msa, scale_msa, gate_msa = self.poseLN_modulation(module_input).chunk(3, dim=-1)

            # Adaptive layer normalization and modulation.
            pose_tokens_modulated = gate_msa * modulate(self.adaln_norm(pose_tokens), shift_msa, scale_msa)
            pose_tokens_modulated = pose_tokens_modulated + pose_tokens

            pose_tokens_modulated = self.trunk(pose_tokens_modulated)
            # Compute the delta update for the pose encoding.
            trunk_features = self.trunk_norm(pose_tokens_modulated)
            if self.split_pose_branch:
                pred_pose_enc_delta = self._split_pose_delta(trunk_features)
            else:
                pred_pose_enc_delta = self.pose_branch(trunk_features)

            if pred_pose_enc is None:
                pred_pose_enc = pred_pose_enc_delta
            else:
                pred_pose_enc = pred_pose_enc + pred_pose_enc_delta

            # Apply final activation functions for translation, rotation, and field-of-view.
            activated_pose = activate_pose(
                pred_pose_enc, trans_act=self.trans_act, quat_act=self.quat_act, fl_act=self.fl_act,
                pose_encoding_type=self.pose_encoding_type
            )
            pred_pose_enc_list.append(activated_pose)

        return pred_pose_enc_list

    def _split_pose_delta(self, trunk_features: torch.Tensor) -> torch.Tensor:
        trans_fov_delta = self.pose_branch_trans_fov(trunk_features)
        rot_delta = self.pose_branch_rot(trunk_features)

        if self.pose_encoding_type in {"absT_quaR_FoV", "absT_eulerR_FoV"}:
            trans_delta = trans_fov_delta[..., :self.trans_dim]
            fov_delta = trans_fov_delta[..., self.trans_dim:]
            return torch.cat([trans_delta, rot_delta, fov_delta], dim=-1)
        if self.pose_encoding_type == "absT_quaR":
            return torch.cat([trans_fov_delta, rot_delta], dim=-1)
        raise NotImplementedError(f"split_pose_branch is not implemented for {self.pose_encoding_type}")


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Modulate the input tensor using scaling and shifting parameters.
    """
    # modified from https://github.com/facebookresearch/DiT/blob/796c29e532f47bba17c5b9c5eb39b9354b8b7c64/models.py#L19
    return x * (1 + scale) + shift


def _pose_component_dims(pose_encoding_type: str) -> tuple[int, int, int]:
    if pose_encoding_type == "absT_quaR_FoV":
        return 3, 4, 2
    if pose_encoding_type == "absT_quaR":
        return 3, 4, 0
    if pose_encoding_type == "absT_eulerR_FoV":
        return 3, 3, 2
    if pose_encoding_type == "quatR":
        return 0, 4, 0
    raise NotImplementedError(f"Unknown pose_encoding_type: {pose_encoding_type}")
