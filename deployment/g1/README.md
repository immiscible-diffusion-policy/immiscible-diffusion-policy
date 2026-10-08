# G1 deployment: Fruit-to-Plate and Can-to-Bin

This optional module is adapted from the research workspace's `g1_dp` deployment path for the banana and soda tasks. It uses **image-based DP**, with either vanilla or immiscible-trained weights. Noise assignment is a training operation; deployment uses the same ordinary policy sampler for both methods. No hand steering, rejection sampling, or mode quotas are exposed by this runner.

The release contains the inference wrapper, asynchronous action scheduling, G1 adapter, task pose/configuration templates, and a synthetic mock loop. It does not contain demonstrations, trained weights, local camera/robot calibration, the low-level controller stack, or a DP3-on-G1 implementation. This module does not reproduce the paper's real-robot results on its own.

## Install

From the release root:

```bash
conda env create -f environments/g1.yaml
conda activate idp-g1
python -m pip install -e . -e backends/dp -e deployment/g1
```

The G1 recipe targets Python 3.10 / PyTorch 2.4.1. The core and bundled DP backend are installed into this environment; the original research checkout is not needed. Real hardware additionally needs the dependencies described below. The Conda recipe has not been solved from scratch; software validation reused installed dependencies.

## What a compatible checkpoint contains

The loader expects the research checkpoint dictionary with `cfg` and `state_dicts`. The configuration must instantiate `diffusion_policy.policy.diffusion_unet_image_policy.DiffusionUnetImagePolicy`, its `MultiImageObsEncoder`, and ResNet encoders. These classes are now included in `backends/dp`. They differ from the hybrid image policy in the Push-T example.

Weights are loaded from `ema_model` when `cfg.training.use_ema` is true, otherwise `model`. Normalization parameters must be included in those weights. The loader instantiates only `cfg.policy`; the old `g1_dp.workspace` and dataset are not imported. Use a checkpoint you trust: loading its pickle/Hydra configuration executes Python. Point-cloud DP3 and the supplied Push-T checkpoints do not satisfy this deployment contract.

The task JSON supplies the frame rate, grasp dimensions, and initial pose. The Python API also accepts an original `data_meta.json` through `metadata_path`. This avoids silently losing grasp semantics when a checkpoint is moved without its training directory.

## Observation and action contract

| Item | Convention |
|---|---|
| Cameras | `ego_cam` (head) and `wide_cam` (external view); H×W×3 uint8 **BGR**, matching the original OpenCV recordings |
| Camera preprocessing | Resize with `INTER_AREA` to checkpoint resolution; CHW float32 divided by 255; checkpoint normalizer and encoder handle subsequent normalization/cropping |
| Paper-task training resolution | 240×320 per camera; read actual resolution from checkpoint |
| State and action | 16 values: left arm 7, right arm 7, left grasp, right grasp |
| Arm order, each side | Shoulder pitch, shoulder roll, shoulder yaw, elbow, wrist roll, wrist pitch, wrist yaw |
| Action units | Absolute joint targets in radians; grasp commands are 0=open and 1=closed |
| Grasp state | Last commanded grasp, not measured finger angles |
| Task frame rate | 30 Hz for banana and soda |
| Timing and sampler | Read observation horizon, action horizon, latency offset, and diffusion steps from the checkpoint; original paper-task deployment uses DDPM |

Do not convert these BGR images to RGB merely because the encoder config calls them `rgb` inputs. Keep camera placement, view order, and image preprocessing consistent with training. The provided JSON files contain reference initial poses from the experiment scripts, not universally calibrated poses for every G1 setup.

`AsyncPolicy` runs one inference at a time in a background thread. At loop frame `f`, it reads prediction index `n_obs_steps - 1 + f - f_obs` from the most recent available chunk. A four-frame blend handles chunk changes; grasp hysteresis uses 0.45/0.55 thresholds. There is no action while a valid chunk is unavailable, so the controller retains its target. The frame convention assumes the control loop maintains its configured rate; validate actual timing on your hardware.

## Check without a robot

Run the automated tests, including a temporary randomly initialized checkpoint and a synthetic episode:

```bash
python -m unittest discover -s deployment/g1/tests -v
```

With your own compatible checkpoint, test loading and prediction without camera/DDS connections:

```bash
python -m idp_g1.run_policy \
  --checkpoint /path/to/model.ckpt \
  --task-config deployment/g1/configs/banana.json \
  --device cuda:0 --check-checkpoint
```

Exercise the whole loop with synthetic camera frames and simulated joint tracking:

```bash
python -m idp_g1.run_policy \
  --checkpoint /path/to/model.ckpt \
  --task-config deployment/g1/configs/banana.json \
  --device cuda:0 --mock --seconds 3 --no-video
```

Use `configs/soda.json` for Can-to-Bin. Synthetic observations validate software plumbing only; they provide no evidence of task success. Generated recordings go under `outputs/g1/`, which is ignored by Git.

## External hardware dependencies

The G1 adapter preserves the experiment interface to the `openpi_g1_control` stack vendored in `unified-finetuning-manager`. Set `G1_CONTROLLER_DIR` to that stack's **robot_control directory**, containing `robot_arm.py` and `dex3_hand.py`. Install the compatible `unitree_sdk2py` SDK and its CycloneDDS runtime in this environment. A stock Unitree SDK install alone does not provide these two controller modules.

Required controller API:

- `G1_29_ArmController`, `G1_29_JointIndex`, and `G1_29_JointArmIndex` from `robot_arm.py`.
- Arm constructor options `motion_mode`, `control_waist`, `simulation_mode`, `dds_already_initialized`, and `dds_interface`; arm position reads, `ctrl_dual_arm(..., use_gravity_compensation=...)`, `speed_gradual_max()`, and the original gravity-compensation/control-thread fields used by the adapter.
- `Dex3DirectController` with `ctrl_dual_hand(left7, right7)` and `stop()`.
- ZMQ image server publishing a **single JPEG message** containing `[ego | wide]` side by side, with the configured head-panel width. The server is external; camera drivers and calibration depend on the user's setup.
- A matching G1 URDF with the 14 named arm joints and finite lower/upper limits. Meshes are not required for reading these limits.

[SOURCE.json](SOURCE.json) records the controller file fingerprints inspected during extraction. No redistributable license was found for those custom controller modules in the inspected checkout, so they are not vendored here. They must be obtained separately or replaced with an implementation of the documented interface; hardware execution remains dependent on that stack. The repository is not a standalone G1 controller distribution.

## Configure and run on G1

Set paths for your installation:

```bash
export G1_CONTROLLER_DIR=/path/to/openpi_g1_control/robot_control
export G1_REMOTE_POSE=/path/to/local/g1_remote_pose.json
```

Record the stationary pose held by your robot's internal controller **before taking arm control**. This helper subscribes to state and publishes no commands:

```bash
python -m idp_g1.capture_pose \
  --iface YOUR_ROBOT_NETWORK_INTERFACE --out "$G1_REMOTE_POSE"
```

Copy and edit the connection template:

```bash
cp deployment/g1/configs/robot.example.json deployment/g1/configs/robot.local.json
```

Set the actual network interface, camera-server address/port, and panel width. `robot.local.json` is ignored by Git. If your network requires explicit DDS peer discovery, set `G1DP_DDS_PEER` to the robot's address; otherwise the SDK's discovery configuration is retained.

Run from an interactive terminal after checking the camera views, joint mapping, calibration, and existing robot controller operation:

```bash
python -m idp_g1.run_policy \
  --checkpoint /path/to/model.ckpt \
  --task-config deployment/g1/configs/banana.json \
  --robot-config deployment/g1/configs/robot.local.json \
  --urdf /path/to/g1.urdf \
  --device cuda:0 --seconds 30
```

Keys: **i** moves to the configured initial pose; **r** starts an episode; **s** stops inference and holds the target; **q** exits. Each episode is capped by `--seconds`. A `STOP` file inside the printed session output directory also ends the session. This is a software hold, not a replacement for the robot's hardware stop.

The runner clips arm commands to URDF limits with a 0.02-radian margin and limits each joint's change to 0.08 radians per tick by default. It rejects non-finite actions. The adapter retains the experiment's hold-on-connect behavior, gravity compensation, and binary-to-Dex3 mapping. Missing camera frames raise an error rather than being reused indefinitely.

On ordinary exit, the original adapter behavior parks the arms at the task initial pose and leaves arm-SDK control held. **Exiting can therefore move the arms.** With `--release-on-exit`, it instead returns to the saved internal-controller pose and ramps control back. That option requires valid local calibration. Process termination, robot balance, and recovery must be handled by the external hardware stack and operator; this release has not been tested on a connected G1.

## Scope and validation

The copied model's `conditional_sample` and `predict_action` methods retain the research implementation. Its training loss now calls the shared assignment helper. The wrapper retains observation padding, resizing, action indexing, blending, and grasp hysteresis. Release changes remove machine-specific paths, require explicit metadata/calibration, surface background inference failures, and simplify the episode loop to ordinary sampling. Experimental hand steering, automatic cycling, teleoperation, data conversion, and collection diagnostics were omitted.

Tests cover real model serialization/inference with generated weights, assignment activation in the compatibility policy, BGR preprocessing, grasp feedback/hysteresis, delayed-chunk indexing, command limits, URDF ordering, worker failures, and mock loop recording. These checks use no real dataset, pretrained checkpoint, camera server, or robot. See [VALIDATION.md](../../VALIDATION.md) for the tested environments and limitations.
