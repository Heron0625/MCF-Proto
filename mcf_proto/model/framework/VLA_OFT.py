"""
VLA-OFT Framework

A lightweight VLA framework that uses action special tokens to predict
continuous actions in parallel, conditioned on multi-view images and a
language instruction.

Key design:
  - Qwen-VL vision-language backbone
  - Action special tokens injected into the sequence
  - Action head (MCFProtoActionHead or MLP) predicts from action token hidden states
"""
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import numpy as np
from PIL import Image

from mcf_proto.training.trainer_utils import initialize_overwatch
from mcf_proto.model.tools import FRAMEWORK_REGISTRY
from mcf_proto.model.framework.base_framework import baseframework
from mcf_proto.model.modules.vlm import get_vlm_model
from mcf_proto.model.modules.action_model.MLP_ActionHeader import get_action_model
from mcf_proto.training.trainer_utils.trainer_tools import resize_images

logger = initialize_overwatch(__name__)

IGNORE_INDEX = -100


@FRAMEWORK_REGISTRY.register("VLA_OFT")
class VLA_OFT(baseframework):
    """
    Vision-Language-Action model with OFT-style action prediction.

    Backbone: Qwen2.5-VL or Qwen3-VL.
    Action head: MCFProtoActionHead (default) or MLP regression head.
    """

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__()
        self.config = config
        self.qwen_vl_interface = get_vlm_model(config=self.config)
        config.framework.action_model.action_hidden_dim = self.qwen_vl_interface.model.config.hidden_size
        self.action_model = get_action_model(config=self.config)

        self.future_action_window_size = config.framework.action_model.future_action_window_size
        self.past_action_window_size = config.framework.action_model.past_action_window_size
        self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size

        self.action_token = "🔍"
        self.action_token_id = self.qwen_vl_interface.processor.tokenizer(
            "🔍", add_special_tokens=False
        )["input_ids"][0]

        self.use_state = config.framework.action_model.get("use_state", False)
        self.l1_loss = nn.L1Loss()

    def _state_to_discrete_text(self, state) -> str:
        """Encode continuous state as a discrete 0-255 integer text string."""
        if state is None:
            return ""
        if isinstance(state, torch.Tensor):
            state_np = state.detach().cpu().numpy().astype(np.float32)
        else:
            state_np = np.array(state, dtype=np.float32)
        if state_np.ndim != 1:
            state_np = state_np.reshape(-1)
        bins = np.linspace(-1, 1, 257, dtype=np.float32)[:-1]
        disc = np.clip(np.digitize(state_np, bins=bins) - 1, 0, 255).astype(np.int32)
        return " ".join(map(str, disc.tolist()))

    def _build_instruction_with_state(self, instruction: str, state) -> str:
        if state is None:
            return instruction
        state_text = self._state_to_discrete_text(state)
        return f"{instruction} Current robot state: {state_text}." if state_text else instruction

    def forward(self, examples: List[dict] = None, **kwargs) -> dict:
        """
        Training forward pass.

        Args:
            examples: list of dicts, each with keys:
                - image: List[PIL.Image]
                - lang:  str
                - action: np.ndarray [T, action_dim]
                - state (optional): np.ndarray or tensor

        Returns:
            dict with 'action_loss' and optional 'loss_dict'
        """
        batch_images = [ex["image"] for ex in examples]
        raw_instructions = [ex["lang"] for ex in examples]
        actions = [ex["action"] for ex in examples]

        if self.use_state:
            states = [ex.get("state", None) for ex in examples]
            instructions = [
                self._build_instruction_with_state(lang, state)
                for lang, state in zip(raw_instructions, states)
            ]
        else:
            instructions = raw_instructions

        action_tokens = self.action_token * self.chunk_len
        prompt_suffix = (
            f" Please predict the next {self.chunk_len} robot actions: "
            f"<action>{action_tokens}</action>."
        )
        instructions = [instr + prompt_suffix for instr in instructions]

        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images, instructions=instructions
        )

        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = qwenvl_outputs.hidden_states[-1]  # [B, L, H]

        with torch.autocast("cuda", dtype=torch.float32):
            input_ids = qwen_inputs.get("input_ids", None)
            action_queries = self._gather_action_token_embeddings(
                last_hidden, input_ids, action_token_id=self.action_token_id
            )  # [B, chunk_len, H]

            actions = torch.tensor(
                np.array(actions), device=action_queries.device, dtype=action_queries.dtype
            )
            actions_target = actions[:, -self.chunk_len:, :]

            action_model_type = self.config.framework.action_model.action_model_type
            if action_model_type == "mcf_proto":
                pred_actions, aux_dict = self.action_model.predict_action(
                    action_queries, return_aux=True
                )

                all_losses = self.action_model.compute_losses(
                    pred_actions, actions_target, aux_dict, time_mask=None
                )

                loss_weights_cfg = self.config.framework.action_model.get("loss_weights", {})
                defaults = {"ortho": 1.0e-2, "smooth": 1.0e-3, "rotation_weight": 1.0}
                lw = {**defaults, **loss_weights_cfg}

                l1_loss = all_losses["l1"]
                rot_weight = float(lw.get("rotation_weight", 1.0))
                reg_loss = (
                    all_losses.get("ortho", torch.tensor(0.0, device=l1_loss.device)) * lw.get("ortho", 0.0)
                    + all_losses.get("smooth", torch.tensor(0.0, device=l1_loss.device)) * lw.get("smooth", 0.0)
                    + all_losses.get("rot", torch.tensor(0.0, device=l1_loss.device)) * rot_weight
                )
                action_loss = l1_loss + reg_loss

                loss_dict = {"l1": l1_loss.detach()}
                for key in ["ortho", "smooth", "rot"]:
                    if key in all_losses:
                        loss_dict[key] = all_losses[key].detach()

                aux_det = {}
                for k, v in aux_dict.items():
                    if isinstance(v, list):
                        aux_det[k] = [x.detach() if isinstance(x, torch.Tensor) else x for x in v]
                    else:
                        aux_det[k] = v.detach() if isinstance(v, torch.Tensor) else v

                if "delta_xyz_mcf" in aux_det and len(aux_det["delta_xyz_mcf"]) > 0:
                    dxm = aux_det["delta_xyz_mcf"][0]
                    dxw = aux_det["delta_xyz_world"][0]
                    drm = aux_det["delta_rpy_mcf"][0]
                    drp = aux_det["delta_rpy_world_pred"][0]
                    loss_dict["norm_delta_xyz_mcf"] = (dxm ** 2).sum(-1).sqrt().mean().detach()
                    loss_dict["norm_delta_xyz_world"] = (dxw ** 2).sum(-1).sqrt().mean().detach()
                    loss_dict["norm_delta_rpy_mcf"] = (drm ** 2).sum(-1).sqrt().mean().detach()
                    loss_dict["delta_rpy_pred_mean"] = drp.mean().detach()
                    loss_dict["delta_rpy_pred_std"] = drp.std().detach()

                return {"action_loss": action_loss, "loss_dict": loss_dict, "aux": aux_det}
            else:
                pred_actions = self.action_model.predict_action(action_queries)
                action_loss = self.l1_loss(pred_actions, actions_target)
                return {"action_loss": action_loss}

    @torch.inference_mode()
    def predict_action(self, examples: List[dict] = None, **kwargs) -> dict:
        """
        Inference forward pass.

        Args:
            examples: same format as forward()

        Returns:
            dict with 'normalized_actions' [B, T, action_dim] and optional 'aux'
        """
        batch_images = [self._to_pil(ex["image"]) for ex in examples]
        raw_instructions = [ex["lang"] for ex in examples]

        train_obs_image_size = getattr(self.config.datasets.vla_data, "image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        if self.use_state:
            states = [ex.get("state", None) for ex in examples]
            instructions = [
                self._build_instruction_with_state(lang, state)
                for lang, state in zip(raw_instructions, states)
            ]
        else:
            instructions = raw_instructions

        action_tokens = self.action_token * self.chunk_len
        prompt_suffix = (
            f" Please predict the next {self.chunk_len} robot actions: "
            f"<action>{action_tokens}</action>."
        )
        instructions = [instr + prompt_suffix for instr in instructions]

        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images, instructions=instructions
        )

        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = qwenvl_outputs.hidden_states[-1]

        with torch.autocast("cuda", dtype=torch.float32):
            input_ids = qwen_inputs.get("input_ids", None)
            action_queries = self._gather_action_token_embeddings(
                last_hidden, input_ids, action_token_id=self.action_token_id
            )

            action_model_type = self.config.framework.action_model.action_model_type
            if action_model_type == "mcf_proto":
                pred_actions, aux_dict = self.action_model.predict_action(
                    action_queries, return_aux=True
                )
                aux_det = {}
                for k, v in aux_dict.items():
                    if isinstance(v, list):
                        aux_det[k] = [
                            x.detach().float().cpu().numpy() if isinstance(x, torch.Tensor) else x
                            for x in v
                        ]
                    else:
                        aux_det[k] = (
                            v.detach().float().cpu().numpy() if isinstance(v, torch.Tensor) else v
                        )
            else:
                pred_actions = self.action_model.predict_action(action_queries)
                aux_det = {}

        normalized_actions = pred_actions.detach().float().cpu().numpy()
        out = {"normalized_actions": normalized_actions}
        if aux_det:
            out["aux"] = aux_det
        return out

    @staticmethod
    def _to_pil(images):
        """Convert images to PIL format if needed."""
        result = []
        for img in images:
            if isinstance(img, Image.Image):
                result.append(img)
            elif isinstance(img, np.ndarray):
                result.append(Image.fromarray(img.astype(np.uint8)))
            else:
                result.append(img)
        return result

    def _gather_action_token_embeddings(
        self,
        last_hidden: torch.Tensor,
        input_ids: torch.Tensor,
        action_token_id=None,
    ) -> torch.Tensor:
        """Extract action token hidden states from the sequence."""
        if action_token_id is None:
            raise ValueError("action_token_id must be provided")

        device = input_ids.device
        B, L, H = last_hidden.shape

        if isinstance(action_token_id, (list, tuple, set)):
            id_list = torch.tensor(list(action_token_id), device=device, dtype=input_ids.dtype)
            mask = torch.isin(input_ids, id_list)
        else:
            mask = (input_ids == action_token_id)

        counts = mask.sum(dim=1)
        if (counts < self.chunk_len).any():
            bad = (counts < self.chunk_len).nonzero(as_tuple=False).flatten().tolist()
            raise RuntimeError(
                f"Samples {bad} have fewer action tokens than chunk_len={self.chunk_len}"
            )

        idx = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
        masked_pos = torch.where(mask, idx, torch.full_like(idx, -1))
        topk_pos = masked_pos.topk(k=self.chunk_len, dim=-1).values
        selected_pos = topk_pos.sort(dim=-1).values
        expanded_index = selected_pos.unsqueeze(-1).expand(-1, -1, H)
        return last_hidden.gather(dim=1, index=expanded_index)
