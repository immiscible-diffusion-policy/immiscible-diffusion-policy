# Integrated 3D Diffusion Policy

DP3 is included with Immiscible Diffusion already integrated. This directory contains the PointNet encoder, conditional UNet, scheduler configuration, normalizers, replay dataset, training loop, checkpoint handling, and inference/offline evaluation. No external DP3 source checkout or patch application is needed.

## Install

From the release root:

```bash
conda env create -f environments/dp3.yaml
conda activate idp-dp3
python -m pip install --no-deps -e .
python -m pip install --no-deps -e backends/dp3
python scripts/smoke_backend.py dp3
```

The recipe targets Linux, Python 3.10, PyTorch 2.4.1, and CUDA 12.1. CPU execution works too. It records versions used in the tested environment; a fresh Conda solve was not run. A compatible existing environment can use the installation commands alone. Training and offline inference do not require RoboTwin, SAPIEN, or PyTorch3D.

## Data

Supply your own Zarr v2 replay buffer:

```text
dataset.zarr/
├── data/
│   ├── action       float32 [N, Da], raw action vectors
│   ├── state        float32 [N, Ds], robot proprioception
│   └── point_cloud  float32 [N, P, C], xyz first, optional rgb in channels 3:6
└── meta/
    └── episode_ends int64 [episodes], cumulative exclusive frame indices
```

All arrays share `N` frames; the final episode end equals `N`. Defaults are `Da=14`, `Ds=14`, `P=1024`, `C=6`. Only xyz is used by default (`policy.use_pc_color=false`). The loader fits and saves the normalizer. Point-cloud coordinates must use a consistent frame, units, point count, and preprocessing across training and inference.

No dataset or raw-sensor preprocessing pipeline is bundled. Existing RoboTwin replay buffers with this schema can be supplied directly. For another robot, update `task.shape_meta.action.shape`, `task.shape_meta.obs.agent_pos.shape`, and `task.shape_meta.obs.point_cloud.shape` to match your arrays.

## Train

Run from `backends/dp3`:

```bash
python train_dp3.py --config-name=immiscible_dp3 \
  dataset_path=/absolute/path/to/robot_dataset.zarr \
  task_name=my_task training.device=cuda:0 \
  training.seed=42 training.num_epochs=600 \
  hydra.run.dir=outputs/my_task_immiscible_s42
```

For vanilla training, add `policy.use_immiscible_diffusion=false` and use a different output directory. The base `robot_dp3` config defaults to vanilla. Set warm-up with `policy.immiscible_start_epoch=50`. Both backends use the release flag `use_immiscible_diffusion`.

The bundled vanilla and Immiscible presets both use **DDPM, epsilon prediction, and 100 denoising steps**. This is an explicit release configuration choice matching the ancestral-sampling recipe in the research workspace. The pinned upstream base used DDIM/sample prediction; the optional upstream patch retains those upstream defaults. The bundled methods share the same sampler, and the policy inference implementation is unchanged.

This is a training integration, not an exact recipe for a paper task. Custom RoboTwin resets, cameras, mirroring, and modality counters are not bundled. These are separate from noise assignment.

Local `logs.json.txt` records losses, epoch, and assignment activation. Checkpoints are saved in the selected run's `checkpoints/` directory. Reuse that directory with `training.resume=true` to resume `latest.ckpt`; `training.num_epochs` is the total target. Model, EMA, optimizer, epoch and update counters are restored. RNG/DataLoader state is not restored, so bitwise continuation is not claimed.

## Offline evaluation

```bash
python eval.py \
  --checkpoint outputs/my_task_immiscible_s42/checkpoints/latest.ckpt \
  --dataset /absolute/path/to/robot_dataset.zarr --device cuda:0
```

This samples action chunks on validation episodes and reports action MSE/MAE in raw units. It does not measure task success or multimodal coverage. Use `--split train` for a training-set diagnostic, `--max-batches` for a short check, or `--raw` for raw weights instead of EMA.

## Simulator or robot integration

Use the trained policy with your environment:

```python
import torch
from immiscible_diffusion_policy.checkpoint import load_policy

policy, config, epoch = load_policy("path/to/latest.ckpt", device="cuda:0")
# obs tensors on cuda:0, in the same units as the training dataset:
# point_cloud: [B, n_obs_steps, P, C]
# agent_pos:   [B, n_obs_steps, Ds]
with torch.no_grad():
    actions = policy.predict_action(obs)["action"]  # [B, n_action_steps, Da]
```

Your environment builds the observation history and executes the action chunk. The copied `RobotRunner` contains observation-history helpers; its upstream `run()` is not a simulator evaluator. **Full RoboTwin rollout evaluation needs the external simulator, assets, and a task adapter**, which are not bundled. No paper success-rate reproduction is claimed.

## Source and release changes

Base: `RoboTwin-Platform/RoboTwin` at `958a6d2910a0262f5531fcdeb7fffae4184bb586`, whose policy derives from Yanjie Ze's 3D Diffusion Policy. [UPSTREAM.json](UPSTREAM.json) records copied files and hashes.

Changes: shared assignment, epoch plumbing, portable dataset/output paths, local logging, method presets, packaging, explicit DDPM configuration, and corrected resume/gradient-accumulation/EMA bookkeeping. An unused torchvision import and unsupported `use_point_crop` scheduler argument were removed. The RoboTwin-specific checkpoint locator was replaced by the portable loader above. Sampling and action-prediction methods are unchanged. See [validation](../../VALIDATION.md) and [third-party notices](../../THIRD_PARTY_NOTICES.md).
