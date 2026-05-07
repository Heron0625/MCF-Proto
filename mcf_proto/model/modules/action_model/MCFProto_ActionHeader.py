"""
MCF-Proto Action Header

Implements MCFProtoActionHead: an action head for VLA policies that combines
a Motion-Centric Action Frame (MCF) with prototype-based action composition.

Key design:
  1. MCF Head: predicts R_t in SO(3) via 6D rotation parameterization
  2. Prototype action generation in MCF:
     - Translation: delta_xyz_mcf = sum_k pi_k * B_trans_k @ z_trans
     - Rotation:    delta_rpy_mcf = sum_k pi_k * B_rot_k   @ z_rot
  3. Map back to world frame:
     - delta_xyz_world = R_t @ delta_xyz_mcf
     - delta_rpy_world = R_t @ delta_rpy_mcf  (small-angle)
  4. Losses: L_act (L1/SmoothL1), L_ortho (prototype orthogonality),
             L_smooth (temporal frame smoothness), L_ent (gating entropy)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import logging

ROT_PI = math.pi

try:
    from omegaconf import ListConfig
except ImportError:
    ListConfig = None

logger = logging.getLogger(__name__)


def rot6d_to_matrix(r6d: torch.Tensor) -> torch.Tensor:
    """Convert 6D rotation representation to 3x3 rotation matrix."""
    a1, a2, a3 = r6d[..., 0], r6d[..., 1], r6d[..., 2]
    b1, b2, b3 = r6d[..., 3], r6d[..., 4], r6d[..., 5]

    a = torch.stack([a1, a2, a3], dim=-1)
    a = F.normalize(a, p=2, dim=-1)

    b = torch.stack([b1, b2, b3], dim=-1)
    b = b - (a * b).sum(dim=-1, keepdim=True) * a
    b = F.normalize(b, p=2, dim=-1)

    c = torch.cross(a, b, dim=-1)
    return torch.stack([a, b, c], dim=-1)  # [..., 3, 3]


def apply_rotation_to_vector(R: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Apply rotation matrix R to vectors v: R @ v (batch-compatible)."""
    return torch.einsum('...ij,...j->...i', R, v)


class MCFProtoActionHead(nn.Module):
    """
    Motion-Centric Frame (MCF) + Prototype Action Head.

    At each timestep:
      1. Predicts R_t in SO(3) to define the Motion-Centric Action Frame.
      2. Generates translation and rotation actions in MCF via prototype composition.
      3. Maps actions back to world frame for supervision.

    Supports multiple MCF groups (e.g. for bimanual arms) and shared/per-group
    prototype dictionaries.
    """

    def __init__(
        self,
        input_dim: int = 2048,
        action_dim: int = 14,
        past_action_window_size: int = 7,
        future_action_window_size: int = 0,
        mcf_groups: Optional[List[Dict]] = None,
        share_dictionary_across_groups: bool = True,
        num_trans_prototypes: int = 16,   # Kt
        num_rot_prototypes: int = 16,     # Kr
        d: int = 3,                       # subspace dimension
        share_R: bool = True,
        share_gating: bool = True,
        share_proto_gating: bool = True,
        share_z: Optional[bool] = None,
        hidden_dim: Optional[int] = None,
        entropy_mode: str = "encourage_sharp",
        rotation_output_space: str = "normalized",
        rotation_rpy_range: Optional[Tuple[float, float]] = None,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.chunk_len = past_action_window_size + 1 + future_action_window_size
        self.Kt = num_trans_prototypes
        self.Kr = num_rot_prototypes
        self.K_gate = max(self.Kt, self.Kr)
        self.d = d
        self.eps = eps
        self.entropy_mode = entropy_mode
        self.share_R = share_R
        self.share_gating = share_gating
        self.share_proto_gating = share_proto_gating
        self.rotation_output_space = rotation_output_space

        self.normalize_eps = 1e-6

        # Rotation normalization range: default [-pi, pi]; overridable from dataset stats
        if rotation_rpy_range is not None:
            mn, mx = float(rotation_rpy_range[0]), float(rotation_rpy_range[1])
            self.register_buffer("rotation_rpy_min", torch.tensor([mn, mn, mn], dtype=torch.float32))
            self.register_buffer("rotation_rpy_max", torch.tensor([mx, mx, mx], dtype=torch.float32))
        else:
            self.register_buffer("rotation_rpy_min", torch.tensor([-ROT_PI] * 3, dtype=torch.float32))
            self.register_buffer("rotation_rpy_max", torch.tensor([ROT_PI] * 3, dtype=torch.float32))

        # Parse MCF groups
        self.mcf_groups = []
        self.mcf_group_slices = []
        if mcf_groups is not None:
            for group in mcf_groups:
                name = group.get("name", "unknown")
                slice_spec = group.get("slice", None)
                if slice_spec is None:
                    continue
                if ListConfig is not None and isinstance(slice_spec, ListConfig):
                    slice_spec = list(slice_spec)
                if isinstance(slice_spec, (list, tuple)) and len(slice_spec) == 2:
                    start, end = int(slice_spec[0]), int(slice_spec[1])
                    if end - start != 6 or start < 0 or end > action_dim:
                        continue
                    self.mcf_groups.append({"name": name, "slice": (start, end)})
                    self.mcf_group_slices.append((start, end))

        self.num_groups = len(self.mcf_groups)
        self.use_mcf = self.num_groups > 0
        if not self.use_mcf:
            logger.warning("No valid MCF groups found. Falling back to pure residual (MLP) head.")

        if hidden_dim is None:
            hidden_dim = input_dim * 2

        # MCF rotation head(s): shared or per-group
        if self.use_mcf:
            head_modules = []
            head_idx_map = {}
            share_R_map = {}
            for g_idx in range(self.num_groups):
                share_key = "shared" if share_R else f"group_{g_idx}"
                if share_key not in head_idx_map:
                    head_idx = len(head_modules)
                    head_modules.append(nn.Sequential(
                        nn.LayerNorm(input_dim),
                        nn.Linear(input_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, 6),
                    ))
                    head_idx_map[share_key] = head_idx
                share_R_map[g_idx] = head_idx_map[share_key]
            self.rotation_heads = nn.ModuleList(head_modules)
            self.rotation_head_map = share_R_map

        # Prototype dictionaries for translation and rotation
        if self.use_mcf:
            if share_dictionary_across_groups:
                self.register_parameter(
                    "trans_prototypes",
                    nn.Parameter(torch.randn(self.Kt, 3, d) * 0.1)
                )
                self.register_parameter(
                    "rot_prototypes",
                    nn.Parameter(torch.randn(self.Kr, 3, d) * 0.1)
                )
            else:
                self.register_parameter(
                    "trans_prototypes",
                    nn.Parameter(torch.randn(self.num_groups, self.Kt, 3, d) * 0.1)
                )
                self.register_parameter(
                    "rot_prototypes",
                    nn.Parameter(torch.randn(self.num_groups, self.Kr, 3, d) * 0.1)
                )
            self.share_dict = share_dictionary_across_groups

        # Gating head(s)
        if self.use_mcf:
            if share_gating:
                self.gating_head = nn.Sequential(
                    nn.LayerNorm(input_dim),
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, self.K_gate),
                )
                self.gating_heads = None
                self.gating_head_rot = None if share_proto_gating else nn.Sequential(
                    nn.LayerNorm(input_dim),
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, self.Kr),
                )
            else:
                self.gating_head = None
                self.gating_head_rot = None
                self.gating_heads = nn.ModuleList([
                    nn.Sequential(
                        nn.LayerNorm(input_dim),
                        nn.Linear(input_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, self.K_gate),
                    ) for _ in range(self.num_groups)
                ])
                self.gating_heads_rot = None if share_proto_gating else nn.ModuleList([
                    nn.Sequential(
                        nn.LayerNorm(input_dim),
                        nn.Linear(input_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, self.Kr),
                    ) for _ in range(self.num_groups)
                ])

        # Subspace coordinate heads (z_trans, z_rot)
        if self.use_mcf:
            if share_z is None:
                share_z = share_gating
            if share_z:
                self.subspace_coord_head_trans = nn.Sequential(
                    nn.LayerNorm(input_dim),
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, d),
                )
                self.subspace_coord_head_rot = nn.Sequential(
                    nn.LayerNorm(input_dim),
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, d),
                )
                self.subspace_coord_heads_trans = None
                self.subspace_coord_heads_rot = None
            else:
                self.subspace_coord_head_trans = None
                self.subspace_coord_head_rot = None
                self.subspace_coord_heads_trans = nn.ModuleList([
                    nn.Sequential(
                        nn.LayerNorm(input_dim),
                        nn.Linear(input_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, d),
                    ) for _ in range(self.num_groups)
                ])
                self.subspace_coord_heads_rot = nn.ModuleList([
                    nn.Sequential(
                        nn.LayerNorm(input_dim),
                        nn.Linear(input_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, d),
                    ) for _ in range(self.num_groups)
                ])

        # Residual head for non-MCF action dimensions
        residual_indices = self._compute_residual_indices()
        self.residual_dim = len(residual_indices)
        self.residual_head = (
            nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.residual_dim),
            ) if self.residual_dim > 0 else None
        )
        self.residual_indices = residual_indices

    # --- Prototype normalization (parameter-free) ---

    def _normalize_prototypes(self, prototypes: torch.Tensor) -> torch.Tensor:
        """L2-normalize each column vector of a prototype dictionary (no learnable params).

        Args:
            prototypes: [K, 3, d] or [G, K, 3, d]
        Returns:
            Same shape; each 3D basis vector normalized to unit length.
        """
        # Normalize along the spatial dim (axis=-2, i.e. the '3' in [K,3,d])
        norm = torch.norm(prototypes, p=2, dim=-2, keepdim=True).clamp(min=self.normalize_eps)
        return prototypes / norm

    # --- Rotation normalization helpers ---

    def _norm_to_rad(self, norm: torch.Tensor) -> torch.Tensor:
        """Map normalized [-1,1] rotation to radians using dataset range."""
        rng = (self.rotation_rpy_max - self.rotation_rpy_min).to(norm.device, norm.dtype)
        return (norm + 1.0) * 0.5 * rng + self.rotation_rpy_min.to(norm.device, norm.dtype)

    def _rad_to_norm(self, rad: torch.Tensor) -> torch.Tensor:
        """Map radians to normalized [-1,1] using dataset range."""
        rng = (self.rotation_rpy_max - self.rotation_rpy_min).to(rad.device, rad.dtype).clamp(min=1e-9)
        return ((rad - self.rotation_rpy_min.to(rad.device, rad.dtype)) / rng * 2.0 - 1.0).clamp(-1.0, 1.0)

    def set_rotation_rpy_range_from_stats(self, low_rot: list, high_rot: list) -> None:
        """Inject rotation range from dataset statistics (q01/q99) for consistent norm/unnorm."""
        dev = self.rotation_rpy_min.device
        low = torch.tensor(low_rot, dtype=torch.float32, device=dev)
        high = torch.tensor(high_rot, dtype=torch.float32, device=dev)
        high = torch.maximum(high, low + 1e-6)
        self.rotation_rpy_min.copy_(low)
        self.rotation_rpy_max.copy_(high)
        logger.info(f"MCFProto: set rotation_rpy range from dataset stats: low={low.tolist()}, high={high.tolist()}")

    def _compute_residual_indices(self) -> List[int]:
        if not self.use_mcf:
            return list(range(self.action_dim))
        covered = set()
        for start, end in self.mcf_group_slices:
            covered.update(range(start, end))
        return [i for i in range(self.action_dim) if i not in covered]

    def _adapt_input_shape(self, h: torch.Tensor) -> torch.Tensor:
        if h.dim() == 2:
            B, H = h.shape
            h = h.unsqueeze(1)
            if not self.training:
                h = h.expand(B, self.chunk_len, H)
        return h

    # --- Forward ---

    def predict_action(
        self,
        actions_hidden_states: torch.Tensor,
        return_aux: bool = False,
        **kwargs,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """
        Predict actions from backbone hidden states.

        Args:
            actions_hidden_states: [B, T, H] or [B, H]
            return_aux: if True, return auxiliary info for loss computation

        Returns:
            pred_actions: [B, T, action_dim]
            aux (if return_aux): dict with R, pi, delta_xyz_mcf, etc.
        """
        h = self._adapt_input_shape(actions_hidden_states)
        B, T, H = h.shape

        pred_actions = torch.zeros(B, T, self.action_dim, device=h.device, dtype=h.dtype)
        aux_dict = {}

        if self.use_mcf:
            for g_idx, group in enumerate(self.mcf_groups):
                start, end = group["slice"]

                R_head = self.rotation_heads[self.rotation_head_map[g_idx]]
                gating_head = self.gating_head if self.gating_head is not None else self.gating_heads[g_idx]
                coord_trans = self.subspace_coord_head_trans if self.subspace_coord_head_trans is not None else self.subspace_coord_heads_trans[g_idx]
                coord_rot = self.subspace_coord_head_rot if self.subspace_coord_head_rot is not None else self.subspace_coord_heads_rot[g_idx]

                # Predict MCF rotation R_t in SO(3)
                R = rot6d_to_matrix(R_head(h))  # [B, T, 3, 3]

                # Gating distributions
                u = gating_head(h)  # [B, T, K_gate]
                if self.share_proto_gating:
                    pi_t = F.softmax(u[..., :self.Kt], dim=-1)
                    pi_r = F.softmax(u[..., :self.Kr], dim=-1)
                else:
                    pi_t = F.softmax(u[..., :self.Kt], dim=-1)
                    rot_gate = self.gating_head_rot if self.gating_head_rot is not None else self.gating_heads_rot[g_idx]
                    pi_r = F.softmax(rot_gate(h), dim=-1)

                z_trans = coord_trans(h)  # [B, T, d]
                z_rot = coord_rot(h)      # [B, T, d]

                B_trans = self._normalize_prototypes(
                    self.trans_prototypes if self.share_dict else self.trans_prototypes[g_idx]
                )
                B_rot = self._normalize_prototypes(
                    self.rot_prototypes if self.share_dict else self.rot_prototypes[g_idx]
                )

                # Prototype composition in MCF:
                # delta_xyz_mcf = sum_k pi_k * B_trans_k @ z_trans
                tmp_trans = torch.einsum('kmd,btd->btkm', B_trans, z_trans)
                delta_xyz_mcf = torch.einsum('btk,btkm->btm', pi_t, tmp_trans)

                # delta_rpy_mcf = sum_k pi_k * B_rot_k @ z_rot
                tmp_rot = torch.einsum('kmd,btd->btkm', B_rot, z_rot)
                delta_rpy_mcf = torch.einsum('btk,btkm->btm', pi_r, tmp_rot)

                # Map back to world frame
                delta_xyz_world = apply_rotation_to_vector(R, delta_xyz_mcf)
                delta_rpy_world_rad = apply_rotation_to_vector(R, delta_rpy_mcf)

                if self.rotation_output_space == "normalized":
                    delta_rpy_world = self._rad_to_norm(delta_rpy_world_rad)
                else:
                    delta_rpy_world = delta_rpy_world_rad

                pred_actions[:, :, start:start + 3] = delta_xyz_world
                pred_actions[:, :, start + 3:start + 6] = delta_rpy_world

                if return_aux:
                    aux_dict.setdefault("R", []).append(R)
                    aux_dict.setdefault("pi", []).append(pi_t)
                    aux_dict.setdefault("pi_rot", []).append(pi_r)
                    aux_dict.setdefault("delta_xyz_mcf", []).append(delta_xyz_mcf)
                    aux_dict.setdefault("delta_rpy_mcf", []).append(delta_rpy_mcf)
                    aux_dict.setdefault("delta_xyz_world", []).append(delta_xyz_world)
                    aux_dict.setdefault("delta_rpy_world_pred", []).append(delta_rpy_world_rad)

        if self.residual_head is not None:
            residual_pred = self.residual_head(h)
            for idx, orig_idx in enumerate(self.residual_indices):
                pred_actions[:, :, orig_idx] = residual_pred[:, :, idx]

        if return_aux:
            return pred_actions, aux_dict
        return pred_actions

    def compute_losses(
        self,
        pred_actions: torch.Tensor,
        actions_target: torch.Tensor,
        aux_dict: Dict,
        time_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute MCF-Proto losses: L_act, L_ortho, L_smooth, L_ent.

        Args:
            pred_actions:   [B, T, action_dim]
            actions_target: [B, T, action_dim]
            aux_dict:       output from predict_action(return_aux=True)
            time_mask:      optional [B, T] validity mask

        Returns:
            dict of loss tensors keyed by name
        """
        losses = {}

        if time_mask is None:
            time_mask = torch.ones(pred_actions.shape[0], pred_actions.shape[1], device=pred_actions.device)
        time_mask = time_mask.float()
        num_valid = time_mask.sum().clamp(min=1.0)

        # L_act: L1 loss on translation/gripper; SmoothL1 on rotation
        # Translation + gripper: L1
        l1_dim_mask = torch.ones(self.action_dim, device=pred_actions.device, dtype=pred_actions.dtype)
        if self.use_mcf:
            for group in self.mcf_groups:
                start, end = group["slice"]
                for j in range(start + 3, start + 6):
                    if j < self.action_dim:
                        l1_dim_mask[j] = 0.0
        l1_valid_dims = l1_dim_mask.sum().clamp(min=1.0)
        l1_diff = torch.abs(pred_actions - actions_target)
        losses["l1"] = (l1_diff * time_mask.unsqueeze(-1) * l1_dim_mask).sum() / (num_valid * l1_valid_dims)

        if not self.use_mcf:
            return losses

        # Rotation loss: SmoothL1 on delta_rpy in radians
        rot_losses = []
        for g_idx, group in enumerate(self.mcf_groups):
            start = group["slice"][0]
            delta_rpy_gt_raw = actions_target[:, :, start + 3:start + 6]
            if self.rotation_output_space == "normalized":
                delta_rpy_gt_rad = self._norm_to_rad(delta_rpy_gt_raw)
            else:
                delta_rpy_gt_rad = delta_rpy_gt_raw
            delta_rpy_pred = aux_dict["delta_rpy_world_pred"][g_idx]
            rot_loss = F.smooth_l1_loss(delta_rpy_pred, delta_rpy_gt_rad, reduction="none").mean(-1)
            rot_losses.append((rot_loss * time_mask).sum() / num_valid)
        if rot_losses:
            losses["rot"] = sum(rot_losses) / len(rot_losses)

        # L_ortho: encourage orthogonality of prototype dictionaries
        ortho_loss = torch.tensor(0.0, device=pred_actions.device, dtype=pred_actions.dtype)
        if self.share_dict:
            for B_raw, Kx in [(self.trans_prototypes, self.Kt), (self.rot_prototypes, self.Kr)]:
                B_dict = self._normalize_prototypes(B_raw)
                BtB = torch.einsum('kmd,kmn->kdn', B_dict, B_dict)
                I = torch.eye(self.d, device=B_dict.device, dtype=B_dict.dtype).unsqueeze(0)
                ortho_loss = ortho_loss + (BtB - I).pow(2).sum() / (Kx * self.d * self.d)
            losses["ortho"] = ortho_loss / 2
        else:
            ortho_losses = []
            for g_idx in range(self.num_groups):
                o = torch.tensor(0.0, device=pred_actions.device, dtype=pred_actions.dtype)
                for B_raw, Kx in [(self.trans_prototypes[g_idx], self.Kt), (self.rot_prototypes[g_idx], self.Kr)]:
                    B_dict = self._normalize_prototypes(B_raw)
                    BtB = torch.einsum('kmd,kmn->kdn', B_dict, B_dict)
                    I = torch.eye(self.d, device=B_dict.device, dtype=B_dict.dtype).unsqueeze(0)
                    o = o + (BtB - I).pow(2).sum() / (Kx * self.d * self.d)
                ortho_losses.append(o / 2)
            losses["ortho"] = sum(ortho_losses) / len(ortho_losses)

        # L_smooth: temporal smoothness of MCF rotation (1 - cos(theta))
        smooth_losses = []
        for R in aux_dict.get("R", []):
            B_sz, T = R.shape[0], R.shape[1]
            if T <= 1:
                continue
            R_t = R[:, 1:]
            R_tm1_T = R[:, :-1].transpose(-2, -1)
            Delta_R = torch.matmul(R_t, R_tm1_T)
            trace = Delta_R.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
            cos_theta = ((trace - 1.0) / 2.0).clamp(-1.0 + self.eps, 1.0 - self.eps)
            valid_pairs = time_mask[:, 1:] * time_mask[:, :-1]
            n_valid_pairs = valid_pairs.sum().clamp(min=1.0)
            smooth_losses.append(((1.0 - cos_theta) * valid_pairs).sum() / n_valid_pairs)
        losses["smooth"] = sum(smooth_losses) / len(smooth_losses) if smooth_losses else torch.tensor(0.0, device=pred_actions.device, dtype=pred_actions.dtype)

        return losses

    def forward(self, actions_hidden_states: torch.Tensor, **kwargs):
        return self.predict_action(actions_hidden_states, **kwargs)
