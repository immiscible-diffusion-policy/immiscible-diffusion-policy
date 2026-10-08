# DP3 integration

**Optional external integration.** The [bundled backend](../../backends/dp3/README.md) already includes the method and needs no patch. Apply this patch only to the pinned external checkout described below.

This patch targets the DP3 implementation bundled with [RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin), pinned to commit `958a6d2910a0262f5531fcdeb7fffae4184bb586`. It is **not** a patch for an arbitrary revision of the standalone 3D Diffusion Policy repository.

It modifies `policy/DP3/3D-Diffusion-Policy/` in three places: the policy, `train_dp3.py`, and `diffusion_policy_3d/config/robot_dp3.yaml`.

## Apply to a separate checkout

Install the backend dependencies following the pinned RoboTwin instructions. In that environment:

```bash
IDP_RELEASE=/absolute/path/to/immiscible-diffusion-policy
python -m pip install --no-deps -e "$IDP_RELEASE"
git clone https://github.com/RoboTwin-Platform/RoboTwin.git robotwin-idp
cd robotwin-idp
git checkout 958a6d2910a0262f5531fcdeb7fffae4184bb586
git apply --check "$IDP_RELEASE/integrations/dp3/immiscible.patch"
git apply "$IDP_RELEASE/integrations/dp3/immiscible.patch"
python -m pip install --no-deps -e policy/DP3/3D-Diffusion-Policy
```

Apply from the **RoboTwin root**, not the nested DP3 directory. The target checkout should be clean. [base.json](base.json) records hashes of every patch-target file.

## Enable during training

In your working `train_dp3.py` command, append these Hydra overrides:

```text
policy.use_immiscible_diffusion=True policy.immiscible_start_epoch=50
```

Alternatively, edit those two fields in `diffusion_policy_3d/config/robot_dp3.yaml` before using the upstream launcher. The pinned shell launcher does not forward arbitrary trailing overrides, so adding them to the shell script invocation alone will not enable assignment.

Task names, dataset preparation, and the remaining launch arguments follow the pinned backend's instructions. No task data is included. The epoch value above is an example, not a claimed reproduction of a specific paper task.

## What changes

- Match normalized action chunks to the already sampled noise pool, then use the assigned noise in the original corruption step.
- Pass `epoch=self.epoch` from both training and validation so the switch follows the resumed workspace epoch.
- Preserve the configured prediction type: epsilon prediction targets the assigned noise, while clean-sample prediction targets the clean trajectory. The patch leaves the upstream velocity-prediction branch untouched; that branch is not validated by this release.
- Preserve the backend's architecture, scheduler, masks, and inference code. This patch does not switch DDIM to DDPM or change denoising steps.

The release uses `use_immiscible_diffusion` consistently across both integrations. The original research workspace's DP3 flag was named `use_immiscible`; use the release flag with this patch. When calling the policy directly, pass `policy.compute_loss(batch, epoch=epoch)`.

This is a method integration, not the complete modified RoboTwin environment used in the paper. Custom tasks, camera settings, augmentation, and evaluation code are not bundled. See [validation](../../VALIDATION.md).
