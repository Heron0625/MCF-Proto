"""
Training script for MCF-Proto VLA model.

Trainer built on PyTorch + Accelerate + DeepSpeed.
Usage:
    accelerate launch --config_file mcf_proto/config/deepseeds/deepspeed_zero2.yaml \\
        --num_processes 4 \\
        mcf_proto/training/train.py \\
        --config_yaml examples/LIBERO/train_files/mcf_proto_train_libero.yaml \\
        [--key value ...]
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Tuple

import numpy as np
import time
import torch
import torch.distributed as dist
import wandb
import yaml
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import AutoProcessor, get_scheduler

from mcf_proto.training.trainer_utils.trainer_tools import normalize_dotlist_args
from mcf_proto.model.framework import build_framework
from mcf_proto.training.trainer_utils.trainer_tools import TrainerUtils
from mcf_proto.training.trainer_utils.trainer_tools import build_param_lr_groups
from mcf_proto.training.trainer_utils.config_tracker import wrap_config, AccessTrackedConfig

deepspeed_plugin = DeepSpeedPlugin()
accelerator = Accelerator(deepspeed_plugin=deepspeed_plugin)
accelerator.print(accelerator.state)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from accelerate.logging import get_logger
logger = get_logger(__name__)


def setup_directories(cfg) -> Path:
    cfg.output_dir = os.path.join(cfg.run_root_dir, cfg.run_id)
    output_dir = Path(cfg.output_dir)
    if not dist.is_initialized() or dist.get_rank() == 0:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir / "checkpoints", exist_ok=True)
    return output_dir


def build_model(cfg) -> torch.nn.Module:
    logger.info(f"Loading Base VLM from: {cfg.framework.qwenvl.base_vlm}")
    return build_framework(cfg)


from mcf_proto.dataloader import build_dataloader


def prepare_data(cfg, accelerator, output_dir):
    logger.info(f"Creating VLA Dataset with Mixture `{cfg.datasets.vla_data.data_mix}`")
    out = build_dataloader(cfg=cfg, dataset_py=cfg.datasets.vla_data.dataset_py)
    vla_train_dataloader = out[0] if isinstance(out, tuple) else out
    accelerator.dataloader_config.dispatch_batches = False
    dist.barrier()
    return vla_train_dataloader


def inject_rotation_range_from_dataset(cfg, model, output_dir: Path) -> None:
    """Inject rotation range from dataset statistics so norm/unnorm is consistent."""
    if not cfg.framework.action_model.get("rotation_range_from_stats", False):
        return
    action_model = getattr(model, "action_model", None)
    if action_model is None or not hasattr(action_model, "set_rotation_rpy_range_from_stats"):
        return
    stats_path = Path(output_dir) / "dataset_statistics.json"
    if not stats_path.exists():
        logger.warning("rotation_range_from_stats=True but dataset_statistics.json not found.")
        return
    try:
        with open(stats_path, "r") as f:
            dataset_statistics = json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load dataset_statistics.json: {e}")
        return
    if not dataset_statistics:
        return
    unnorm_key = next(iter(dataset_statistics.keys()))
    action_stats = dataset_statistics.get(unnorm_key, {}).get("action", {})
    q01 = action_stats.get("q01")
    q99 = action_stats.get("q99")
    if q01 is None or q99 is None or len(q01) < 6 or len(q99) < 6:
        logger.warning("dataset_statistics action missing q01/q99. Rotation range left at default.")
        return
    low_rot = [float(q01[3]), float(q01[4]), float(q01[5])]
    high_rot = [float(q99[3]), float(q99[4]), float(q99[5])]
    action_model.set_rotation_rpy_range_from_stats(low_rot, high_rot)
    logger.info("Injected rotation range from dataset statistics.")


def setup_optimizer_and_scheduler(model, cfg):
    param_groups = build_param_lr_groups(model=model, cfg=cfg)
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=cfg.trainer.learning_rate.base,
        betas=tuple(cfg.trainer.optimizer.betas),
        weight_decay=cfg.trainer.optimizer.weight_decay,
        eps=cfg.trainer.optimizer.eps,
    )
    lr_scheduler = get_scheduler(
        name=cfg.trainer.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=cfg.trainer.num_warmup_steps,
        num_training_steps=cfg.trainer.max_train_steps,
        scheduler_specific_kwargs=cfg.trainer.scheduler_specific_kwargs,
    )
    return optimizer, lr_scheduler


class VLATrainer(TrainerUtils):
    def __init__(self, cfg, model, vla_train_dataloader, optimizer, lr_scheduler, accelerator):
        self.config = cfg
        self.model = model
        self.vla_train_dataloader = vla_train_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.accelerator = accelerator
        self.completed_steps = 0
        self.total_batch_size = (
            cfg.datasets.vla_data.per_device_batch_size
            * accelerator.num_processes
            * accelerator.gradient_accumulation_steps
        )

    def prepare_training(self):
        rank = dist.get_rank() if dist.is_initialized() else 0
        seed = self.config.seed + rank if hasattr(self.config, "seed") else rank + 42
        set_seed(seed)
        self._init_checkpointing()
        self._adjust_lr_scheduler_for_resume()
        freeze_modules = (
            self.config.trainer.freeze_modules
            if hasattr(self.config.trainer, "freeze_modules") else None
        )
        self.model = self.freeze_backbones(self.model, freeze_modules=freeze_modules)
        self.print_trainable_parameters(self.model)
        self.model, self.optimizer, self.vla_train_dataloader = self.setup_distributed_training(
            self.accelerator, self.model, self.optimizer, self.vla_train_dataloader,
        )
        self._load_optimizer_state()
        self._init_wandb()

    def _adjust_lr_scheduler_for_resume(self):
        if self.completed_steps > 0:
            for _ in range(self.completed_steps):
                self.lr_scheduler.step()

    def _init_wandb(self):
        if self.accelerator.is_main_process:
            wandb.init(
                name=self.config.run_id,
                dir=os.path.join(self.config.output_dir, "wandb"),
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                group="vla-train",
            )

    def _init_checkpointing(self):
        self.checkpoint_dir = os.path.join(self.config.output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.optimizer_ckpt_base = None
        pretrained_checkpoint = getattr(self.config.trainer, "pretrained_checkpoint", None)
        is_resume = getattr(self.config.trainer, "is_resume", False)
        self.resume_from_checkpoint = pretrained_checkpoint
        if is_resume:
            resume_from_checkpoint, self.completed_steps = self._get_latest_checkpoint(self.checkpoint_dir)
            if resume_from_checkpoint:
                self.resume_from_checkpoint = resume_from_checkpoint
                self.model = self.load_pretrained_backbones(self.model, self.resume_from_checkpoint, reload_modules=None)
                if getattr(self.config.trainer, "save_optimizer_state", False):
                    self.optimizer_ckpt_base = resume_from_checkpoint.replace("_pytorch_model.pt", "")
                logger.info(f"Resuming from checkpoint: {self.resume_from_checkpoint}, step {self.completed_steps}")
                return
            else:
                logger.warning(f"No checkpoint found in {self.checkpoint_dir}. Starting from scratch.")
                self.completed_steps = 0
        if pretrained_checkpoint:
            reload_modules = getattr(self.config.trainer, "reload_modules", None)
            self.model = self.load_pretrained_backbones(self.model, pretrained_checkpoint, reload_modules=reload_modules)
            try:
                self.completed_steps = int(re.search(r"steps_(\d+)_pytorch_model\.pt", pretrained_checkpoint).group(1))
            except AttributeError:
                self.completed_steps = 0
        else:
            self.completed_steps = 0

    def _load_optimizer_state(self):
        if not getattr(self.config.trainer, "save_optimizer_state", False):
            return
        if not getattr(self, "optimizer_ckpt_base", None):
            return
        num_ranks = dist.get_world_size() if dist.is_initialized() else 1
        all_states = []
        for r in range(num_ranks):
            path = self.optimizer_ckpt_base + f"_optimizer_rank{r}.pt"
            if not os.path.exists(path):
                logger.warning(f"Optimizer state not found for rank {r}. Optimizer starts fresh.")
                return
            all_states.append(torch.load(path, map_location="cpu", weights_only=False))
        self.optimizer.load_state_dict(all_states)

    def _save_checkpoint(self):
        if self.accelerator.is_main_process:
            checkpoint_path = os.path.join(self.checkpoint_dir, f"steps_{self.completed_steps}")
            state_dict = self.accelerator.get_state_dict(self.model)
            torch.save(state_dict, checkpoint_path + "_pytorch_model.pt")
            with open(os.path.join(self.config.output_dir, "summary.jsonl"), "a") as f:
                f.write(json.dumps({"steps": self.completed_steps}) + "\n")
            if isinstance(self.config, AccessTrackedConfig):
                self.config.save_accessed_config(Path(self.config.output_dir) / "config.yaml", use_original_values=False)
        if getattr(self.config.trainer, "save_optimizer_state", False):
            rank = dist.get_rank() if dist.is_initialized() else 0
            checkpoint_path = os.path.join(self.checkpoint_dir, f"steps_{self.completed_steps}")
            torch.save(self.optimizer.state_dict(), checkpoint_path + f"_optimizer_rank{rank}.pt")
        self.accelerator.wait_for_everyone()

    def _log_metrics(self, metrics):
        if self.completed_steps % self.config.trainer.logging_frequency == 0:
            if dist.get_rank() == 0:
                metrics["learning_rate"] = self.lr_scheduler.get_last_lr()[0]
                metrics["epoch"] = round(self.completed_steps / len(self.vla_train_dataloader), 2)
                wandb.log(metrics, step=self.completed_steps)
                items = ", ".join(
                    f"'{k}': {v:.5f}" if isinstance(v, float) else f"'{k}': {v}"
                    for k, v in sorted(metrics.items())
                )
                logger.info(f"Step {self.completed_steps}: {{ {items} }}")

    def _create_data_iterators(self):
        self.vla_iter = iter(self.vla_train_dataloader)

    def _get_next_batch(self):
        try:
            return next(self.vla_iter)
        except StopIteration:
            if not hasattr(self, "vla_epoch_count"):
                self.vla_epoch_count = 0
            self.vla_iter, self.vla_epoch_count = TrainerUtils._reset_dataloader(
                self.vla_train_dataloader, self.vla_epoch_count
            )
            return next(self.vla_iter)

    def _train_step(self, batch_vla):
        with self.accelerator.accumulate(self.model):
            self.optimizer.zero_grad()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output_dict = self.model.forward(batch_vla)
                action_loss = output_dict["action_loss"]
            self.accelerator.backward(action_loss)
            if self.config.trainer.gradient_clipping is not None:
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.trainer.gradient_clipping)
            self.optimizer.step()
            self.lr_scheduler.step()
        metrics = {"action_loss": action_loss.item()}
        if "loss_dict" in output_dict:
            for k, v in output_dict["loss_dict"].items():
                metrics[f"loss_{k}"] = v.item() if isinstance(v, torch.Tensor) else float(v)
        return metrics

    def eval_action_model(self, step_metrics: dict) -> dict:
        examples = self._get_next_batch()
        actions = np.array([ex["action"] for ex in examples])
        output_dict = self.model.predict_action(examples=examples)
        if self.accelerator.is_main_process:
            normalized_actions = output_dict["normalized_actions"]
            step_metrics["mse_score"] = TrainerUtils.euclidean_distance(normalized_actions, actions) / np.prod(actions.shape)
        del examples
        dist.barrier()
        return step_metrics

    def train(self):
        if self.accelerator.is_main_process:
            logger.info("***** Training Configuration *****")
            logger.info(f"  Total steps = {self.config.trainer.max_train_steps}")
            logger.info(f"  Batch size per device = {self.config.datasets.vla_data.per_device_batch_size}")
            logger.info(f"  Global batch size = {self.total_batch_size}")
        self._create_data_iterators()
        progress_bar = tqdm(
            range(self.config.trainer.max_train_steps),
            initial=self.completed_steps,
            disable=not self.accelerator.is_local_main_process,
        )
        while self.completed_steps < self.config.trainer.max_train_steps:
            t0 = time.perf_counter()
            batch_vla = self._get_next_batch()
            t1 = time.perf_counter()
            step_metrics = self._train_step(batch_vla)
            t2 = time.perf_counter()
            if self.accelerator.sync_gradients:
                progress_bar.update(1)
                self.completed_steps += 1
            if self.accelerator.is_local_main_process:
                progress_bar.set_postfix({"data": f"{t1-t0:.3f}", "model": f"{t2-t1:.3f}"})
            if self.completed_steps % self.config.trainer.eval_interval == 0:
                step_metrics = self.eval_action_model(step_metrics)
            step_metrics["data_time"] = t1 - t0
            step_metrics["model_time"] = t2 - t1
            self._log_metrics(step_metrics)
            if self.completed_steps % self.config.trainer.save_interval == 0 and self.completed_steps > 0:
                self._save_checkpoint()
            if self.completed_steps >= self.config.trainer.max_train_steps:
                break
        self._finalize_training()

    def _finalize_training(self):
        if self.accelerator.is_main_process:
            final_checkpoint = os.path.join(self.config.output_dir, "final_model")
            os.makedirs(final_checkpoint, exist_ok=True)
            state_dict = self.accelerator.get_state_dict(self.model)
            torch.save(state_dict, os.path.join(final_checkpoint, "pytorch_model.pt"))
            logger.info(f"Training complete. Final model saved at {final_checkpoint}")
        if self.accelerator.is_main_process:
            wandb.finish()
        self.accelerator.wait_for_everyone()


def main(cfg) -> None:
    logger.info("MCF-Proto VLA Training :: Starting")
    cfg = wrap_config(cfg)
    output_dir = setup_directories(cfg=cfg)
    vla = build_framework(cfg)
    vla_train_dataloader = prepare_data(cfg=cfg, accelerator=accelerator, output_dir=output_dir)
    inject_rotation_range_from_dataset(cfg, vla, output_dir)
    optimizer, lr_scheduler = setup_optimizer_and_scheduler(model=vla, cfg=cfg)
    trainer = VLATrainer(
        cfg=cfg,
        model=vla,
        vla_train_dataloader=vla_train_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
    )
    trainer.prepare_training()
    trainer.train()
    logger.info("Training complete.")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, required=True, help="Path to YAML config")
    args, clipargs = parser.parse_known_args()
    cfg = OmegaConf.load(args.config_yaml)
    dotlist = normalize_dotlist_args(clipargs)
    cli_cfg = OmegaConf.from_dotlist(dotlist)
    cfg = OmegaConf.merge(cfg, cli_cfg)
    main(cfg)
