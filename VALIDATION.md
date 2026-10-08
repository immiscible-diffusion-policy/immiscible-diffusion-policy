# Validation record

Checked on October 8, 2026. Validation used CPU execution and temporary environments outside this release directory. No real demonstration dataset, pretrained checkpoint, or robot hardware was used. Bundled-backend checks generated temporary synthetic data/checkpoints and exercised short Push-T simulator rollouts.

## Core package

All **14 unit tests passed** in both environments:

| Python | PyTorch | SciPy | PyYAML |
|---|---|---|---|
| 3.9.18 | 1.12.1 | 1.9.1 | 6.0.2 |
| 3.10.20 | 2.4.1+cu121 | 1.15.3 | 6.0.3 |

These are tested combinations, not a claim that every combination allowed by the package's dependency lower bounds has been exercised. GPU execution was not tested.

Run the portable tests from the release root after installing the core package:

```bash
python -m unittest discover -s tests -v
```

The tests cover:

- Exact assigned-noise agreement with the original experiment formula for multiple seeds, horizons, and action dimensions.
- Global optimality on a small enumerated problem where squared and unsquared costs produce different assignments.
- One-to-one assignment, unchanged input tensors, preserved complete noise rows, and unchanged random-generator state.
- Constant coordinates, all-constant batches, batch size one, float64 output, and rejected invalid inputs.
- Independent pairing before warm-up or when disabled; activation at the specified epoch.
- Correct epsilon and clean-sample targets and finite gradients that update model parameters.

The CPU example was exercised with active assignment, disabled assignment, pre-activation warm-up, and clean-sample prediction. These are single training-step checks, not convergence or task-performance experiments.

Editable installation was checked outside the source directory. A wheel was also built, installed into a separate temporary environment, and imported from that environment's `site-packages`. Its matching operation passed a smoke check, and all bundled license notices were verified inside the wheel. Build products were removed from the release folder afterward.

## Integration patches

All nine unmodified target files were verified byte-for-byte against their public pinned GitHub revisions. Each patch passed `git apply --check`, applied successfully in a temporary directory, and passed reverse-application checks. Target revisions and original file hashes are recorded in the integration `base.json` files.

The patched Python files compiled. AST comparison confirmed that all three policies' `conditional_sample` and `predict_action` methods were unchanged. Both training and validation calls in all three workspaces pass the workspace's current epoch.

A local audit exercised the actual patched `compute_loss` function bodies using small model, encoder, normalizer, and scheduler test doubles. It covered **36 combinations**: three policies × two prediction targets × global/inpainting conditioning × disabled/warm-up/active assignment. Disabled and warm-up paths matched the pinned vanilla function bodies; active full-chunk paths matched the current experiment function bodies, including sampled noise and timesteps. Gradients were finite. Two additional low-dimensional action-window cases passed with the two upstream action-offset conventions.

That audit required the original research workspace and was not bundled. The portable unit tests contain a copied reference of the matching formula; they do not require the workspace.

The optional patch audit above used test doubles; it did not instantiate the full backends. The integrated source release was subsequently exercised with actual networks as described below. DP3 velocity prediction and GPU execution remain untested. No claim is made that the release reproduces paper benchmark results or historical checkpoint behavior.

## Bundled DP and DP3 end-to-end checks

The copied low-dimensional DP, hybrid RGB DP, and point-cloud DP3 were all instantiated and trained on CPU using small generated Zarr replay buffers and reduced UNet widths. Tests ran the actual training entry points, dataset loaders, normalizers, encoders, diffusion losses, optimizers, EMA updates, checkpoint serialization, and policy samplers. These were short smoke checks, not benchmark training runs.

Each check trained epoch 0, saved a checkpoint, resumed for epoch 1, loaded EMA weights for offline inference, and checked finite action errors. Immiscible checks verified assignment disabled at epoch 0 and enabled at epoch 1; vanilla checks verified assignment remained disabled. Saved optimizer/EMA counters and final batch index were verified. Additional DP-lowdim and DP3 checks used an accumulation group larger than the available two batches to exercise the final partial group. Checkpoints restored the counters without resetting EMA warm-up.

Both DP variants completed a short headless Push-T simulator rollout through `eval.py`, producing a finite `test/mean_score`. This validates the evaluation path, not policy competence. No RoboTwin simulator or real-robot rollout was performed. DP3 evaluation is offline only.

Reproduce in the corresponding installed backend environments, from the release root:

```bash
# idp-dp environment
python scripts/smoke_backend.py dp-lowdim
python scripts/smoke_backend.py dp-lowdim --vanilla --rollout
python scripts/smoke_backend.py dp-lowdim --accumulation 3
python scripts/smoke_backend.py dp-rgb --rollout
python scripts/smoke_backend.py dp-rgb --vanilla

# idp-dp3 environment
python scripts/smoke_backend.py dp3
python scripts/smoke_backend.py dp3 --vanilla
python scripts/smoke_backend.py dp3 --accumulation 3
```

The script writes data and weights only under a temporary directory and removes them on exit. It explicitly imports the bundled source rather than any separately installed research checkout. The two backend dependency environments were reused for CPU validation; the supplied Conda recipes were not solved from scratch. GPU training and long-run learning behavior were not evaluated.

Core, DP, and DP3 wheels were built and installed into temporary environments. Imports resolved to the installed copies, both DP policies and the Push-T environment imported successfully, and the backend Hydra configuration files were present in the wheels. These installs reused system dependency packages rather than provisioning fresh Conda stacks.

## Release boundaries

The release contains source, documentation, configuration, tests, patches, citation metadata, and license notices. Datasets, pretrained weights, generated results, original Git histories, and remote configuration are excluded. Original research source Python files were fingerprinted before and after preparation to verify they were unchanged. A fresh local Git repository was initialized on branch `main`, with no commits or remotes; nothing was pushed.

## Optional G1 deployment

Ten deployment tests passed under Python 3.9 / PyTorch 1.12.1 and Python 3.10 / PyTorch 2.4.1. They cover observation padding, BGR resize/channel preservation, command-echo grasp state, grasp hysteresis, delayed-chunk indexing/blending, asynchronous inference and error propagation, finite/limited commands, URDF joint ordering, and avoiding hold commands after a completed controller hand-off.

The integration test builds the actual multi-image ResNet/UNet policy with reduced image size and UNet widths, checks disabled/active assignment at consecutive epochs and finite gradients, serializes random EMA weights in a temporary checkpoint, reloads them through G1DPPolicy, and verifies equal predictions with a fixed RNG. It then runs a short synthetic mock episode and checks its recorded commands. Test data, weights, and recordings are temporary and removed at exit. No real demonstration or pretrained checkpoint was read.

The new compatibility policy's conditional_sample and predict_action methods match the original research source by AST comparison. Their checkpoint architecture is included, but historical G1 checkpoint compatibility has not been tested with an actual paper checkpoint. The supplied G1 dependency recipe was not solved from scratch.

No real camera, DDS connection, robot command, or hardware rollout was attempted. The custom controller modules remain external; their fingerprints and extraction provenance are in deployment/g1/SOURCE.json. The root release still has no commits or remote, and nothing was pushed.

The optional G1 package and updated DP backend were built as wheels in a temporary directory and installed into an isolated virtual environment using existing dependency packages. Imports resolved to the installed release packages rather than the research checkout, and all ten G1 tests passed again there. A before/after fingerprint check confirmed 355 original research source/configuration files were unchanged.
