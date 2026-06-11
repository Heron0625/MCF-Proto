# MCF-Proto: Motion-Centric Action Frames with Prototype Composition for Vision-Language-Action Models

This repository contains the official implementation of **MCF-Proto**, a lightweight action head for Vision-Language-Action (VLA) models that replaces the conventional world-frame action head with a motion-centric representation and prototype-based action composition.

## Overview

Most VLA models predict end-effector actions directly in the world (base) frame, which introduces a fundamental misalignment: the robot's visual observations are body-centric, but the predicted actions are world-centric. MCF-Proto addresses this with two components:

1. **Motion-Centric Action Frame (MCF)**: At each timestep the model predicts a rotation R_t ∈ SO(3) that defines a local frame aligned with the end-effector's current motion direction. Actions are predicted in this frame and then rotated back to the world frame.

2. **Prototype-based Action Composition**: Instead of regressing actions directly, MCF-Proto maintains two learned prototype dictionaries — B_trans (Kt prototypes for translation) and B_rot (Kr prototypes for rotation) — each of shape [K, 3, d]. The final action is a soft-weighted combination of low-dimensional latent codes projected through these prototypes.

### Action Prediction

Given action token embeddings from the VLM backbone:

```
# Translation in MCF
delta_xyz_mcf = sum_k pi_k * B_trans[k] @ z_trans    # [3]

# Rotation in MCF
delta_rpy_mcf = sum_k pi_k * B_rot[k] @ z_rot        # [3]

# Map back to world frame
delta_xyz_world = R_t @ delta_xyz_mcf
delta_rpy_world = R_t @ delta_rpy_mcf
```

### Training Objective

```
L = L_act + lambda_ortho * L_ortho + lambda_smooth * L_smooth
```

- **L_act**: L1 loss for translation/gripper; SmoothL1 loss for rotation (in radians)
- **L_ortho**: Orthogonality regularizer `||B^T B - I||_F^2` for both prototype dictionaries
- **L_smooth**: Temporal smoothness of the MCF rotation `1 - cos(angle(R_t @ R_{t-1}^T))`

## Demo
![demo](out_file1_10m.mp4)
![demo](out_file2_10m.mp4)
![demo](out_file3_10m.mp4)


### Pick tube (x4 speed)
https://github.com/user-attachments/assets/c9ec1ff2-f318-4ffe-b2d5-4db58026ba3d

### Pass cube (x4 speed)
https://github.com/user-attachments/assets/4c5bd832-6ad4-4db9-a29e-1c1e2253bf14

### Pass cube in OOD scenario
https://github.com/user-attachments/assets/9e0747c4-c3e7-4d12-bac9-387c26accc0b



## Installation

```bash
pip install -r requirements.txt
```

Key dependencies:
- Python >= 3.10
- PyTorch >= 2.1.0
- Transformers >= 4.45.0
- Accelerate >= 0.30.0
- DeepSpeed >= 0.14.0
- LeRobot (for data loading)


## Data Preparation

MCF-Proto uses the [LeRobot](https://github.com/huggingface/lerobot) dataset format.

### LIBERO

Download the LIBERO dataset and convert it to LeRobot format. Set `data_root_dir` in the config to the root of the converted dataset. The expected directory structure is:

```
LEROBOT_LIBERO_DATA/
  libero_10/
    episode_0/
      ...
  libero_spatial/
    ...
  ...
```

Set `data_mix: libero_all` to train on all LIBERO tasks, or specify a subset (e.g., `libero_10`).


## Training

### LIBERO

1. Download the Qwen3-VL-4B-Instruct (or Qwen2.5-VL) checkpoint.

2. Edit `examples/LIBERO/train_files/run_libero_train.sh`:
   ```bash
   base_vlm=/path/to/Qwen3-VL-4B-Instruct
   libero_data_root=/path/to/LEROBOT_LIBERO_DATA
   ```

3. Launch training:
   ```bash
   bash examples/LIBERO/train_files/run_libero_train.sh
   ```

This uses Accelerate + DeepSpeed ZeRO-2 across 4 GPUs by default. Adjust `--num_processes` and `per_device_batch_size` to match your hardware.

### Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_trans_prototypes` | 16 | Number of translation prototypes (Kt) |
| `num_rot_prototypes` | 16 | Number of rotation prototypes (Kr) |
| `d` | 3 | Subspace dimension |
| `lambda_ortho` | 1e-2 | Orthogonality loss weight |
| `lambda_smooth` | 1e-3 | Smoothness loss weight |

See `examples/LIBERO/train_files/mcf_proto_train_libero.yaml` for the full configuration.


## Repository Structure

```
mcf_proto/
├── mcf_proto/
│   ├── model/
│   │   ├── framework/
│   │   │   └── VLA_OFT.py          # Main VLA framework (Qwen VLM + MCF-Proto head)
│   │   └── modules/
│   │       ├── action_model/
│   │       │   ├── MCFProto_ActionHeader.py   # Core MCF-Proto action head
│   │       │   └── MLP_ActionHeader.py        # Action model factory
│   │       └── vlm/
│   │           ├── QWen3.py        # Qwen3-VL interface
│   │           └── QWen2_5.py      # Qwen2.5-VL interface
│   ├── dataloader/
│   │   ├── lerobot_datasets.py     # LeRobot dataset wrapper
│   │   └── gr00t_lerobot/          # Data pipeline utilities
│   ├── training/
│   │   └── train.py                # Training entry point
│   └── config/
│       └── deepseeds/              # DeepSpeed / Accelerate configs
└── examples/
    └── LIBERO/
        └── train_files/
            ├── mcf_proto_train_libero.yaml
            └── run_libero_train.sh
```


## Configuration

All training options are controlled via YAML config files. Command-line arguments override YAML values using dot-notation (e.g., `--trainer.learning_rate.base 1e-4`).

The action model section in the YAML:

```yaml
framework:
  action_model:
    action_model_type: mcf_proto
    action_dim: 7
    future_action_window_size: 7
    mcf_groups:
      - {name: ee, slice: [0, 6]}   # end-effector dims (3 trans + 3 rot)
    num_trans_prototypes: 16
    num_rot_prototypes: 16
    d: 3
    rotation_output_space: normalized   # predict rotation in [-1,1], unnorm to radians
    rotation_range_from_stats: true     # infer range from dataset statistics
```


## License

This code is released under the MIT License.
