# Integrated Diffusion Policy

The low-dimensional UNet and hybrid RGB UNet are included with Immiscible Diffusion already integrated. This directory includes the required models, normalizers, replay buffer, sequence sampler, datasets, training workspaces, checkpoint handling, and Push-T evaluation environment. No external source checkout or patch application is needed.

## Install

From the release root:

```bash
conda env create -f environments/dp.yaml
conda activate idp-dp
python -m pip install --no-deps -e .
python -m pip install --no-deps -e backends/dp
python scripts/smoke_backend.py dp-lowdim
python scripts/smoke_backend.py dp-rgb
```

The recipe targets Linux, Python 3.9, PyTorch 1.12.1, and CUDA 11.6. CPU execution also works. It records versions used in the tested environment; a fresh Conda solve was not run. A compatible existing environment can use the two installation commands alone. Avoid installing another `diffusion_policy` package in the same environment. See [validation](../../VALIDATION.md).

## Data

Supply your own Zarr v2 replay buffer. No dataset is included or downloaded automatically.

```text
dataset.zarr/
├── data/
│   ├── action       float32 [N, 2], raw action coordinates
│   ├── state        float32 [N, >=2], pusher position in the first two columns
│   ├── keypoint     float32 [N, 9, 2], required for low-dimensional DP
│   └── img          uint8   [N, 96, 96, 3], required for RGB DP
└── meta/
    └── episode_ends int64 [episodes], cumulative exclusive frame indices
```

All arrays share `N` frames; the final episode end equals `N`. Low-dimensional observations concatenate the flattened keypoints and pusher position. The RGB loader converts images to float32 CHW in `[0,1]`. Supply raw actions: the normalizer is fitted during training and saved with the model. Train/validation splits are made by episode before sampling windows.

For another robot, implement a dataset with the same batch contract, update observation/action dimensions, and disable Push-T rollouts until you supply a suitable environment runner. Push-T evaluation requires Push-T observation/action semantics.

## Train

Run from `backends/dp`, using an absolute dataset path:

```bash
# Low-dimensional Immiscible Diffusion Policy
python train.py --config-name=immiscible_pusht_lowdim \
  task.dataset.zarr_path=/absolute/path/to/pusht.zarr \
  task.dataset.max_train_episodes=null \
  training.device=cuda:0 training.seed=42 training.num_epochs=801 \
  hydra.run.dir=outputs/pusht_lowdim_immiscible_s42

# RGB Immiscible Diffusion Policy
python train.py --config-name=immiscible_pusht_rgb \
  task.dataset.zarr_path=/absolute/path/to/pusht.zarr \
  task.dataset.max_train_episodes=null \
  training.device=cuda:0 training.seed=42 training.num_epochs=801 \
  hydra.run.dir=outputs/pusht_rgb_immiscible_s42
```

For vanilla training, add `policy.use_immiscible_diffusion=false` and choose a **different output directory**. Set activation with `policy.immiscible_start_epoch=50`: assignment starts at zero-based epoch 50, after epochs 0–49. The original base configs (`train_diffusion_unet_lowdim_workspace`, `train_diffusion_unet_hybrid_workspace`) default to vanilla.

These are training examples, not exact paper experiment recipes. They retain upstream Push-T resets and do not include the research workspace's mirrored batches or modality counters. W&B is disabled by default; local logs, configuration, and checkpoints are still written. Add `training.enable_rollouts=false` for offline training without a simulator process.

Reusing an output directory with `training.resume=true` loads `checkpoints/latest.ckpt`. `training.num_epochs` is the total target, including completed epochs. Model, EMA, optimizer, epoch, and optimizer/EMA update counters are restored. RNG/DataLoader state is not restored, so bitwise continuation is not claimed. The final epoch is saved even if it is outside the regular checkpoint interval.

## Evaluate

Offline action prediction on validation episodes:

```bash
python eval_offline.py \
  --checkpoint outputs/pusht_lowdim_immiscible_s42/checkpoints/latest.ckpt \
  --dataset /absolute/path/to/pusht.zarr --device cuda:0
```

This reports stochastic open-loop action MSE/MAE in raw dataset units, not task success or mode coverage. `--raw` selects raw weights instead of EMA; `--max-batches` limits the check.

Push-T simulator evaluation, also supported for the RGB checkpoint:

```bash
SDL_VIDEODRIVER=dummy python eval.py \
  --checkpoint outputs/pusht_lowdim_immiscible_s42/checkpoints/latest.ckpt \
  --output_dir outputs/pusht_lowdim_eval --device cuda:0
```

The checkpoint's environment-runner settings determine rollout counts. Results are written to `eval_log.json`, plus videos when enabled. Offline evaluation does not start a simulator.

Direct inference:

```python
import torch
from immiscible_diffusion_policy.checkpoint import load_policy

policy, config, epoch = load_policy("path/to/latest.ckpt", device="cpu")
# Supply obs using the dataset's units/scaling:
# lowdim: {"obs": tensor[B, n_obs_steps, 20]}
# RGB: {"image": tensor[B, n_obs_steps, 3, 96, 96],
#       "agent_pos": tensor[B, n_obs_steps, 2]}
with torch.no_grad():
    actions = policy.predict_action(obs)["action"]
```

There is no Hungarian assignment at inference time.

## Source and release changes

Base: `real-stanford/diffusion_policy` at `5ba07ac6661db573af695b419a7947ecb704690f`. [UPSTREAM.json](UPSTREAM.json) records copied files and source hashes. Only the dependency closure for the two policies and Push-T is bundled.

Changes: shared assignment, epoch plumbing, method presets, packaging, optional rollouts, float32 image loading, CPU checkpoint loading, synchronous/final-epoch saves, and corrected resume/gradient-accumulation/EMA bookkeeping. Obsolete upstream demo helpers with stale paths were removed. The policy inference methods are unchanged. See [third-party notices](../../THIRD_PARTY_NOTICES.md).

## G1 checkpoint architecture

This backend also includes `DiffusionUnetImagePolicy`, `MultiImageObsEncoder`, and the ResNet factory used by the G1 image-DP checkpoints. This is a different architecture from the hybrid RGB Push-T preset. Its original Hydra targets and inference methods are preserved; its loss calls the shared assignment routine. See [G1 deployment](../../deployment/g1/README.md). These compatibility modules do not add a G1 dataset loader or training preset.
