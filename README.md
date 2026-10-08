# Immiscible Diffusion Policy

Official repository for **Immiscible Diffusion Policy: Preserving Multimodal Robot Actions through Label-Free Noise Assignment**.

**Under review for ICRA 2027.**

[Paper](https://arxiv.org/abs/2610.09369)

## Overview

Robot demonstrations can contain multiple valid ways to complete the same task, such as grasping an object with either hand. Yet a diffusion policy can collapse to one action modality even when the demonstrations are balanced. Our work connects this behavior to mixing between diffusion paths induced by independent action-noise pairing during training.

**Immiscible Diffusion Policy changes how action chunks are paired with Gaussian noise during training.** Within each batch, we:

1. Sample a pool of Gaussian noise and compute distances to the action chunks in a standardized action space.
2. Use a one-to-one Hungarian assignment to minimize the total action-noise distance.
3. Train with the reassigned noise using the existing diffusion objective.

This assignment aims to keep noise-to-action paths more distinct and preserve multiple demonstrated behaviors. It reuses every sampled noise tensor exactly once, requires **no modality labels**, and leaves the **policy architecture and inference procedure unchanged**. The method integrates with both Diffusion Policy (DP) and 3D Diffusion Policy (DP3).

See [Algorithm details](#algorithm-details) for the matching cost, normalization, and training code.

## Implementation

This repository includes the necessary DP and DP3 source code with Immiscible Diffusion already integrated. You can train on your own data without cloning another policy repository or applying a patch. **Datasets and pretrained checkpoints are not included.** This is a method and training-code release, not a full reproduction package for the paper's benchmark tables or robot experiments.

## Choose a starting point

| Goal | Entry point |
|---|---|
| Understand assignment without a dataset | CPU example below |
| Train low-dimensional or RGB DP | [backends/dp](backends/dp/README.md) |
| Train point-cloud DP3 | [backends/dp3](backends/dp3/README.md) |
| Deploy image-based DP on G1 | [deployment/g1](deployment/g1/README.md) |
| Modify an external upstream checkout | [Optional DP patch](integrations/diffusion_policy/README.md), [optional DP3 patch](integrations/dp3/README.md) |

The backends include actual encoders, UNets, datasets, normalization, training, EMA, checkpoints, and inference. DP includes Push-T rollout evaluation. DP3 includes offline evaluation and a policy interface; its RoboTwin simulator adapter is external. An optional [G1 deployment module](deployment/g1/README.md) provides image-DP checkpoint loading, camera/state preprocessing, and the robot adapter for the two real-world tasks; the low-level controller and SDK remain external.

```text
immiscible_diffusion_policy/  shared assignment and checkpoint/evaluation helpers
backends/dp/                integrated lowdim/RGB DP and Push-T evaluation
backends/dp3/               integrated point-cloud DP3 and offline evaluation
deployment/g1/             optional G1 image-DP deployment and configuration
environments/              separate backend dependency recipes
examples/                  minimal CPU training-step demonstration
scripts/smoke_backend.py    temporary-data training/resume/inference checks
integrations/               optional patches for external upstream checkouts
tests/                      portable algorithm tests
```

## Core-only quickstart

Use Python 3.9 or newer. From this repository's root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[examples]'
python examples/training_step.py
python examples/training_step.py --vanilla
python examples/training_step.py --epoch 49
python examples/training_step.py --prediction-type sample
python -m unittest discover -s tests -v
```

The example prints the loss, gradient norm, and whether assignment is active. It writes no data or checkpoint files. Settings are in [configs/example.yaml](configs/example.yaml); use `--config path/to/config.yaml` to supply another file. These are demonstration settings, not paper experiment settings. The tiny network and illustrative schedule demonstrate the training operation, not learned multimodal robot behavior.

For an existing policy environment that already has PyTorch and SciPy, install only the core package with `python -m pip install --no-deps -e /path/to/immiscible-diffusion-policy`. NumPy is supplied through SciPy. The standalone example additionally requires PyYAML.

## Bundled backend installation

From the release root, install the desired backend in its own environment:

```bash
# DP, low-dimensional and RGB
conda env create -f environments/dp.yaml
conda activate idp-dp
python -m pip install --no-deps -e .
python -m pip install --no-deps -e backends/dp
python scripts/smoke_backend.py dp-lowdim
python scripts/smoke_backend.py dp-rgb

# DP3, point clouds (a separate environment)
conda env create -f environments/dp3.yaml
conda activate idp-dp3
python -m pip install --no-deps -e .
python -m pip install --no-deps -e backends/dp3
python scripts/smoke_backend.py dp3
```

Smoke checks create random replay buffers and small training checkpoints in a temporary directory, test training/resume/inference, and then delete the directory. They do not ship or download data, and are not task-performance experiments. Recipes record tested library versions; fresh Conda solves and GPU training have not been validated. Backend READMEs provide data schemas, vanilla/Immiscible launch commands, and evaluation instructions. The optional checkpoint/evaluation helpers in the core package require backend dependencies.

## Algorithm details

Let `actions` be dataset-normalized action chunks of shape `[batch, horizon, action_dim]`. Independently draw one Gaussian noise tensor of the same shape for each chunk. After a configurable warm-up:

1. Flatten each action chunk and noise sample.
2. Standardize each action coordinate across the batch using population standard deviation, clamped to `1e-6`.
3. Compute the **unsquared Euclidean** distance from every standardized action chunk to every noise sample.
4. Find the minimum-total-cost one-to-one assignment using SciPy's Hungarian solver.
5. Permute the original noise pool and use it in the existing forward corruption and loss.

Only the matching cost uses batch-standardized actions. The clean actions passed to the diffusion process retain their original dataset normalization. Every noise sample is used exactly once. Assignment preserves the noise pool while changing the action-noise pairing; it requires no modality labels. It does not guarantee that every trained policy will express every mode.

```python
import torch
from immiscible_diffusion_policy import match_noise

# actions: dataset-normalized [B, T, Da]
noise = torch.randn_like(actions)
if use_immiscible_diffusion and epoch >= immiscible_start_epoch:
    noise = match_noise(actions, noise)

timesteps = torch.randint(
    scheduler.config.num_train_timesteps,
    (actions.shape[0],), device=actions.device,
)
noisy_actions = scheduler.add_noise(actions, noise, timesteps)
# Pass noisy_actions, timesteps and observations to your existing denoiser.
# Epsilon prediction: target = noise (the assigned noise).
# Clean-sample prediction: target = actions.
```

The assignment operation preserves the architecture, observation conditioning, loss masking, and inference procedure. The bundled DP3 configuration explicitly selects DDPM for both methods; the optional upstream patch retains upstream sampler defaults. For policies that jointly denoise action and observation-feature channels, matching uses the action channels and permutes each **complete** noise row. See the bundled policies or optional integration patches for the actual insertion points.

## API and activation

- `match_noise(actions, noise)` returns the permuted noise tensor.
- `assignment_indices(actions, noise)` returns the corresponding permutation on the input device.
- Inputs may have shape `[B, Da]` or `[B, T, Da]`. Noise may have additional trailing feature channels. Batch and time axes must match.
- Matching is computed in float32 without gradients; output noise retains its original dtype and device. Inputs are not modified and no new randomness is consumed.
- The reference pairwise cost uses `O(B² × T × Da)` working memory. SciPy solves on CPU, so GPU matching includes a synchronization and transfer of the cost matrix.

The examples, bundled backends, and patches expose **fixed-epoch activation**, with zero-based epoch indexing: `50` activates when `epoch >= 50`, after epochs 0–49. This reflects the inspected experiment implementations. The paper's performance/modality-based criterion for selecting the activation point is not implemented as an automatic controller. Choose and document the activation epoch for your task. `0` activates immediately; disabling assignment preserves independent pairing.

## Optional upstream patches

| Backend | Scope | Pinned base |
|---|---|---|
| [Diffusion Policy](integrations/diffusion_policy/README.md) | Low-dimensional UNet and hybrid RGB UNet, training/validation calls, configs | `real-stanford/diffusion_policy` at `5ba07ac6661db573af695b419a7947ecb704690f` |
| [DP3](integrations/dp3/README.md) | RoboTwin's DP3 policy, training/validation calls, config | `RoboTwin-Platform/RoboTwin` at `958a6d2910a0262f5531fcdeb7fffae4184bb586` |

These patches are for separate upstream checkouts only; do not apply them to the already integrated backends. They are not snapshots of the paper's full training environment. Mirroring, RoboTwin task resets, and modality counters are outside this release. G1 deployment is documented separately in [deployment/g1](deployment/g1/README.md). Assignment works with the batches supplied by your training pipeline.

## Validation

The core tests check agreement with the experiment assignment formula, one-to-one optimal matching, preservation of complete noise rows, constant coordinates, batch size one, warm-up behavior, prediction targets, and gradient flow. See [VALIDATION.md](VALIDATION.md) for the tested environments, integration checks, and their limits.

## Citation

```bibtex
@article{zhang2026immiscible,
  title={Immiscible Diffusion Policy: Preserving Multimodal Robot Actions through Label-Free Noise Assignment},
  author={Zhang, Xiao and Chen, Yuxin and Liang, Zhixuan and Zhan, Guojian and Li, Chenran and Xu, Chenfeng and Tomizuka, Masayoshi and Li, Yiheng},
  journal={arXiv preprint arXiv:2610.09369},
  year={2026}
}
```

## Acknowledgments and license

The method builds on Immiscible Diffusion's noise-assignment principle and integrates with Diffusion Policy and 3D Diffusion Policy. This release is distributed under the MIT license. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and [licenses/](licenses/) for source provenance and preserved upstream notices.

## Optional G1 deployment

See [deployment/g1/README.md](deployment/g1/README.md) for the Fruit-to-Plate (banana) and Can-to-Bin (soda) image-DP deployment path. It includes task templates, the original checkpoint architecture, asynchronous inference, the G1 adapter, and hardware-free checks. The robot dependencies are optional. No G1 dataset, checkpoint, or robot-specific calibration is distributed. DP3-on-G1 is not implemented or claimed.
