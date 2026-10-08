"""Hardware abstraction for deployment.

Implement `RobotInterface` for the real G1 (SDK joint position control + hand + cameras); everything
else in `idp_g1` (closed-loop runner, replay / open-loop tests) only talks to this interface.
`MockRobot` replays a recorded episode's camera frames and simulates first-order joint tracking so the
whole loop can be exercised without hardware.

Conventions (must match training data, see G1DPPolicy.contract()):
  obs  = {"ego_cam": HxWx3 uint8 BGR, "wide_cam": HxWx3 uint8 BGR, "qpos": (16,) float32, "t": seconds}
         qpos = [left arm 7 | right arm 7 | left grasp | right grasp]; grasp dims may be anything — the
         policy wrapper overwrites them with the last command.
  action = (16,) float32 absolute joint targets (rad) + grasp commands in {0, 1}
"""
from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from typing import Dict, Optional

import numpy as np

ARM_DIMS = list(range(14))
GRASP_DIMS = [14, 15]


class RobotInterface(ABC):
    fps: float = 30.0
    name: str = "robot"

    def connect(self):
        pass

    def close(self):
        pass

    @abstractmethod
    def get_obs(self) -> Dict[str, np.ndarray]:
        ...

    @abstractmethod
    def send_action(self, action: np.ndarray):
        ...

    def home(self, q_target: Optional[np.ndarray] = None, duration: float = 3.0):
        """Default: linear interpolation from the current qpos to q_target over `duration` seconds."""
        q0 = self.get_obs()["qpos"].copy()
        if q_target is None:
            return
        n = max(int(duration * self.fps), 1)
        for i in range(1, n + 1):
            a = q0 + (i / n) * (np.asarray(q_target, dtype=np.float32) - q0)
            a[GRASP_DIMS] = q_target[GRASP_DIMS]
            self.send_action(a)
            time.sleep(1.0 / self.fps)

    def stop(self):
        """Software stop hook: hold the current position; not a hardware emergency stop."""
        try:
            self.send_action(self.get_obs()["qpos"])
        except Exception:
            pass


class SafetyLimiter:
    """Clamp joint targets to limits and to a maximum per-tick change; counts interventions.

    limits: (14, 2) rad from the URDF (with `margin` shrink); max_step_rad: max |Δ| per tick per joint
    relative to the last *sent* command (the data's per-frame change: median 0.02, p95 0.06 rad).
    """

    def __init__(self, limits: np.ndarray, max_step_rad: float = 0.08, margin: float = 0.02):
        limits = np.asarray(limits, dtype=np.float32)
        if limits.shape != (14, 2) or not np.isfinite(limits).all():
            raise ValueError("Expected finite (14, 2) joint limits")
        if not np.isfinite(max_step_rad) or max_step_rad <= 0 or not np.isfinite(margin) or margin < 0:
            raise ValueError("Invalid step limit or joint margin")
        self.lo = limits[:, 0] + margin
        self.hi = limits[:, 1] - margin
        if (self.lo >= self.hi).any():
            raise ValueError("Joint margin leaves an empty interval")
        self.max_step = max_step_rad
        self.last: Optional[np.ndarray] = None
        self.n_clamped_limit = 0
        self.n_clamped_step = 0

    def reset(self, q_current: np.ndarray):
        q = np.asarray(q_current, dtype=np.float32)[:14]
        if q.shape != (14,) or not np.isfinite(q).all():
            raise ValueError("Expected finite measured arm state")
        if ((q < self.lo) | (q > self.hi)).any():
            raise ValueError("Measured joints are outside the configured limits/margin")
        self.last = q.copy()

    def __call__(self, action: np.ndarray) -> np.ndarray:
        a = np.asarray(action, dtype=np.float32).copy()
        if a.shape != (16,) or not np.isfinite(a).all():
            raise ValueError("Expected finite (16,) action")
        arm = a[:14]
        clipped = np.clip(arm, self.lo, self.hi)
        self.n_clamped_limit += int((clipped != arm).sum())
        arm = clipped
        if self.last is not None:
            d = np.clip(arm - self.last, -self.max_step, self.max_step)
            self.n_clamped_step += int((np.abs(arm - self.last) > self.max_step).sum())
            arm = self.last + d
        self.last = arm.copy()
        a[:14] = arm
        a[GRASP_DIMS] = (a[GRASP_DIMS] > 0.5).astype(np.float32)
        return a

    def stats(self):
        return dict(clamped_limit=self.n_clamped_limit, clamped_step=self.n_clamped_step, max_step_rad=self.max_step)


def urdf_arm_limits(path: str) -> np.ndarray:
    """Read the experiment joint order from the user's G1 URDF; no mesh assets needed."""
    import xml.etree.ElementTree as ET
    root = ET.parse(path).getroot()
    joints = {j.attrib["name"]: j for j in root.findall("joint")}
    order = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
             "wrist_roll", "wrist_pitch", "wrist_yaw")
    limits = []
    for side in ("left", "right"):
        for name in order:
            lim = joints[f"{side}_{name}_joint"].find("limit")
            limits.append([float(lim.attrib["lower"]), float(lim.attrib["upper"])])
    result = np.asarray(limits, dtype=np.float32)
    if result.shape != (14, 2) or not np.isfinite(result).all() or (result[:, 0] >= result[:, 1]).any():
        raise ValueError("Invalid arm joint limits in URDF")
    return result


class MockRobot(RobotInterface):
    """Fake robot: camera frames come from a recorded episode (they do not react to the policy);
    joints follow commands with first-order tracking (alpha per tick ≈ the data's 2–3 frame lag);
    optional observation latency and joint noise."""

    name = "mock"

    def __init__(self, frames: Dict[str, np.ndarray], q0: np.ndarray, fps: float = 30.0, alpha: float = 0.4,
                 obs_latency_frames: int = 0, joint_noise_std: float = 0.0, seed: int = 0, loop_frames: bool = True):
        self.frames = frames
        self.T = next(iter(frames.values())).shape[0]
        self.q = np.asarray(q0, dtype=np.float32).copy()
        self.fps = fps
        self.alpha = alpha
        self.obs_latency = obs_latency_frames
        self.noise = joint_noise_std
        self.rng = np.random.default_rng(seed)
        self.loop_frames = loop_frames
        self.tick = 0
        self.hist = []
        self.log_q, self.log_cmd = [], []

    @classmethod
    def from_zarr_episode(cls, zarr_path: str, episode: int, **kw):
        from diffusion_policy.common.replay_buffer import ReplayBuffer
        rb = ReplayBuffer.create_from_path(zarr_path, mode="r")
        e = rb.get_episode(int(episode))
        frames = {k: e[k] for k in e if k not in ("state", "action")}
        return cls(frames, e["state"][0], **kw), e

    def _frame_index(self):
        i = self.tick
        return (i % self.T) if self.loop_frames else min(i, self.T - 1)

    def get_obs(self):
        i = self._frame_index()
        q = self.q.copy()
        if self.noise > 0:
            q[:14] += self.rng.normal(0, self.noise, 14).astype(np.float32)
        self.hist.append(q)
        j = max(len(self.hist) - 1 - self.obs_latency, 0)
        obs = {k: v[i] for k, v in self.frames.items()}
        obs["qpos"] = self.hist[j]
        obs["t"] = self.tick / self.fps
        return obs

    def send_action(self, action):
        a = np.asarray(action, dtype=np.float32)
        self.q[:14] += self.alpha * (a[:14] - self.q[:14])
        self.q[GRASP_DIMS] = a[GRASP_DIMS]
        self.log_q.append(self.q.copy()); self.log_cmd.append(a.copy())
        self.tick += 1


