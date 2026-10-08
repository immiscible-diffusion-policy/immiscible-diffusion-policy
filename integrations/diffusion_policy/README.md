# Diffusion Policy integration

**Optional external integration.** The [bundled backend](../../backends/dp/README.md) already includes the method and needs no patch. Apply this patch only to the pinned external checkout described below.

This patch adds noise assignment to the **low-dimensional UNet** and **hybrid image UNet** in [Diffusion Policy](https://github.com/real-stanford/diffusion_policy), pinned to commit `5ba07ac6661db573af695b419a7947ecb704690f`. It changes two policy files, two workspaces, and their two training configurations. It does not include the RoboTwin RGB fork or the full paper experiment setup.

## Apply to a separate checkout

First install the backend's dependencies following its pinned README. In that environment, install this release without replacing the backend's dependency versions:

```bash
IDP_RELEASE=/absolute/path/to/immiscible-diffusion-policy
python -m pip install --no-deps -e "$IDP_RELEASE"
git clone https://github.com/real-stanford/diffusion_policy.git diffusion-policy-idp
cd diffusion-policy-idp
git checkout 5ba07ac6661db573af695b419a7947ecb704690f
git apply --check "$IDP_RELEASE/integrations/diffusion_policy/immiscible.patch"
git apply "$IDP_RELEASE/integrations/diffusion_policy/immiscible.patch"
python -m pip install --no-deps -e .
```

`IDP_RELEASE` must point to this release's root. The target checkout should be clean. [base.json](base.json) records hashes of every patch-target file.

## Enable during training

After obtaining and preparing your own task data using the backend's instructions, an example low-dimensional launch is:

```bash
python train.py --config-name=train_diffusion_unet_lowdim_workspace \
  policy.use_immiscible_diffusion=True \
  policy.immiscible_start_epoch=50
```

For the hybrid image policy use `--config-name=train_diffusion_unet_hybrid_workspace` and the appropriate upstream task/data overrides. Set `policy.use_immiscible_diffusion=False` for independent pairing. These commands retain the upstream configurations and are not paper benchmark reproduction commands.

## What changes

- The policy samples noise exactly as before and calls `match_noise` before forward corruption when enabled and `epoch >= immiscible_start_epoch`.
- Both training and validation calls explicitly pass `epoch=self.epoch`, including after resuming the workspace. Validation measures the same active training objective.
- In the low-dimensional policy, matching uses the action channels of the trajectory actually being denoised. This also handles upstream `pred_action_steps_only=True`; the paper's full-chunk setting remains the default.
- Hybrid-image matching uses normalized actions; feature channels, when present, move with the complete assigned noise row.
- The original loss target selection, conditioning mask, architecture, scheduler, and inference methods are preserved. Inference does not need an epoch.

The added constructor options are `use_immiscible_diffusion=False` and `immiscible_start_epoch=50`. If calling `compute_loss` outside the patched workspace, explicitly supply the current epoch: `policy.compute_loss(batch, epoch=epoch)`. Its default is zero.

Mirroring, fixed symmetric resets, modality counters, and task-specific paper configurations are not added by this patch. See [validation](../../VALIDATION.md) for checks performed and limits.
