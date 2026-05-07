#!/bin/bash
# Training script for MCF-Proto on LIBERO.
# === Please modify the following paths before running ===

export NCCL_SOCKET_IFNAME=eth0
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1

Framework_name=VLA_OFT
freeze_module_list=''

# Path to a local Qwen3-VL-4B-Instruct model checkpoint
base_vlm=/path/to/Qwen3-VL-4B-Instruct

config_yaml=./examples/LIBERO/train_files/mcf_proto_train_libero.yaml
libero_data_root=/path/to/LEROBOT_LIBERO_DATA
data_mix=libero_all
run_root_dir=./results/checkpoints
run_id=mcf_proto_libero

# === End of configuration ===

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}

accelerate launch \
  --config_file mcf_proto/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 4 \
  mcf_proto/training/train.py \
  --config_yaml ${config_yaml} \
  --framework.name ${Framework_name} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --datasets.vla_data.data_root_dir ${libero_data_root} \
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.per_device_batch_size 16 \
  --datasets.vla_data.video_backend torchvision_av \
  --trainer.freeze_modules ${freeze_module_list} \
  --trainer.max_train_steps 100000 \
  --trainer.save_interval 20000 \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 100 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project mcf_proto_libero \
  --wandb_entity your_wandb_entity \
  --trainer.is_resume true
