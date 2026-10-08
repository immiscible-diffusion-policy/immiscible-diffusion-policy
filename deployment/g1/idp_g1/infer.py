"""Deployment layer: checkpoint -> G1DPPolicy (+ ChunkScheduler / AsyncPolicy for real-time loops).

Timing model (30 Hz, DDPM100 ≈ 290 ms ≈ 9 frames on an A6000):
  * A chunk predicted from the observation at frame f_obs covers frames f_obs + k, k = 0 .. H-2
    (pred index To-1+k). Actions are always indexed by *frame offset from the observation*, so a
    chunk that arrives late is simply consumed from the right offset — nothing is stale.
  * `n_latency_steps` (training config) documents the expected offset at which execution starts;
    the synchronous `step()` returns pred[To-1+L : To-1+L+n_action_steps].
  * `ChunkScheduler` keeps the newest chunk(s), returns the command for the current frame, blends the
    first `blend_steps` frames of a new chunk with the previous one (they overlap because H > L+Na),
    applies grasp hysteresis and feeds the last grasp *command* back into qpos (the data's grasp
    state is a command echo).  `AsyncPolicy` runs inference in a background thread around it.

Deployment contract: see `G1DPPolicy.contract()`.
"""
from __future__ import annotations

import json
import os
import threading
import time
import warnings
from collections import deque
from typing import Dict, List, Optional

import cv2
import dill
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

from diffusion_policy.common.pytorch_util import dict_apply

OmegaConf.register_new_resolver("eval", eval, replace=True)


def load_policy_from_checkpoint(ckpt_path: str, device: str = "cuda:0", use_ema: Optional[bool] = None):
    payload = torch.load(ckpt_path, pickle_module=dill, map_location="cpu")
    cfg = payload["cfg"]
    if use_ema is None:
        use_ema = bool(cfg.training.use_ema)
    key = "ema_model" if use_ema else "model"
    if key not in payload["state_dicts"]:
        raise KeyError(f"{ckpt_path} has no '{key}' weights (available: {sorted(payload['state_dicts'])}); pass use_ema={not use_ema}")
    policy = hydra.utils.instantiate(cfg.policy)
    policy.load_state_dict(payload["state_dicts"][key])
    policy.to(torch.device(device)).eval()
    return policy, cfg


class ObsBuffer:
    """Last n observations, head-padded by repeating the oldest (training convention)."""

    def __init__(self, n_obs_steps: int):
        self.n = n_obs_steps
        self.buf = deque(maxlen=n_obs_steps)

    def reset(self):
        self.buf.clear()

    def push(self, obs):
        self.buf.append(obs)

    def stacked(self):
        assert len(self.buf) > 0, "push an observation first"
        items = list(self.buf)
        while len(items) < self.n:
            items.insert(0, items[0])
        return {k: np.stack([o[k] for o in items]) for k in items[0]}


class G1DPPolicy:
    def __init__(self, ckpt_path: str, device: str = "cuda:0", num_inference_steps: Optional[int] = None,
                 scheduler: Optional[str] = None, use_ema: Optional[bool] = None,
                 n_latency_steps: Optional[int] = None, n_samples: int = 1, sample_reduce: str = "mean",
                 grasp_threshold: float = 0.5, grasp_hysteresis: float = 0.1, grasp_state_from_command: bool = True,
                 ood_margin: float = 0.15, metadata_path: Optional[str] = None):
        self.policy, self.cfg = load_policy_from_checkpoint(ckpt_path, device, use_ema)
        self.device = torch.device(device)
        shape_meta = OmegaConf.to_container(self.cfg.shape_meta, resolve=True)
        self.rgb_keys = [k for k, v in shape_meta["obs"].items() if v.get("type") == "rgb"]
        self.lowdim_key = [k for k, v in shape_meta["obs"].items() if v.get("type", "low_dim") == "low_dim"][0]
        self.img_hw = {k: tuple(shape_meta["obs"][k]["shape"][1:]) for k in self.rgb_keys}
        self.n_obs_steps = int(self.cfg.n_obs_steps)
        self.horizon = int(self.policy.horizon)
        self.action_dim = int(shape_meta["action"]["shape"][0])
        self.n_latency_steps = int(self.cfg.get("n_latency_steps", 0)) if n_latency_steps is None else int(n_latency_steps)
        self.n_action_steps = int(self.policy.n_action_steps) - int(self.cfg.get("n_latency_steps", 0))
        assert self.n_obs_steps - 1 + self.n_latency_steps + self.n_action_steps <= self.horizon, \
            f"To-1+L+Na = {self.n_obs_steps - 1 + self.n_latency_steps + self.n_action_steps} > horizon {self.horizon}"
        run_dir = os.path.dirname(os.path.dirname(os.path.realpath(ckpt_path)))   # realpath: symlinked ckpts resolve to their run dir
        meta_path = os.path.join(run_dir, "data_meta.json")
        if metadata_path is not None:
            meta_path = os.path.expanduser(metadata_path)
        if not os.path.isfile(meta_path):
            raise FileNotFoundError("Provide metadata_path (deployment configs/<task>.json or original data_meta.json)")
        with open(meta_path) as stream:
            self.data_meta = json.load(stream)
        self.spec = self.data_meta.get("spec", {})
        self.binary_dims = list(self.spec.get("binary_action_dims", []))
        if self.action_dim != 16 or self.binary_dims != [14, 15]:
            raise ValueError("G1 deployment requires action_dim=16 and binary_action_dims=[14, 15]")
        if set(self.rgb_keys) != {"ego_cam", "wide_cam"} or shape_meta["obs"][self.lowdim_key]["shape"] != [16]:
            raise ValueError("Expected ego_cam, wide_cam and a 16-dimensional robot state")
        self.fps = float(self.data_meta["fps"])
        if not np.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("metadata fps must be positive and finite")
        self.n_samples = int(n_samples)
        self.sample_reduce = sample_reduce
        self.grasp_threshold = grasp_threshold
        self.grasp_hysteresis = grasp_hysteresis
        self.grasp_state_from_command = grasp_state_from_command
        self.ood_margin = ood_margin
        stats = self.policy.normalizer[self.lowdim_key].params_dict["input_stats"]
        self.state_min = stats["min"].detach().cpu().numpy(); self.state_max = stats["max"].detach().cpu().numpy()
        self.buffer = ObsBuffer(self.n_obs_steps)
        self.last_grasp = np.zeros(len(self.binary_dims), dtype=np.float32)
        self.last_pred: Optional[np.ndarray] = None
        self.last_ood_dims: List[int] = []
        self._ood_warned = set()
        self.set_sampler(scheduler, num_inference_steps)

    # ---- sampler (inference only) ------------------------------------------------------------
    def set_sampler(self, scheduler: Optional[str], num_inference_steps: Optional[int]):
        from diffusers.schedulers.scheduling_ddim import DDIMScheduler
        from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
        conf = dict(hydra.utils.instantiate(self.cfg.policy.noise_scheduler).config)
        if scheduler is None or scheduler.lower() == "ddpm":
            self.policy.noise_scheduler = DDPMScheduler(**{k: v for k, v in conf.items() if not k.startswith("_")})
        elif scheduler.lower() == "ddim":
            keys = ("num_train_timesteps", "beta_start", "beta_end", "beta_schedule", "clip_sample", "prediction_type")
            self.policy.noise_scheduler = DDIMScheduler(**{k: conf[k] for k in keys if k in conf}, set_alpha_to_one=True, steps_offset=0)
        else:
            raise ValueError(scheduler)
        if num_inference_steps is not None:
            self.policy.num_inference_steps = int(num_inference_steps)
        self.sampler_desc = f"{type(self.policy.noise_scheduler).__name__}({self.policy.num_inference_steps})"

    def warmup(self, n: int = 2):
        """Pay initialization costs before connecting to hardware, then reset history."""
        dummy = {k: np.zeros((*self.img_hw[k], 3), np.uint8) for k in self.rgb_keys}
        dummy["qpos"] = np.zeros_like(self.state_min, dtype=np.float32)
        for _ in range(n):
            self.reset()
            for _ in range(self.n_obs_steps):
                self.buffer.push(self.prep_obs(dummy))
            self.predict(self.buffer.stacked())
        self.reset()

    # ---- observation handling ----------------------------------------------------------------
    def prep_obs(self, obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        out = {}
        for k in self.rgb_keys:
            im = np.asarray(obs[k])
            assert im.ndim == 3 and im.shape[-1] == 3 and im.dtype == np.uint8, f"{k}: expected HxWx3 uint8 BGR, got {im.shape} {im.dtype}"
            H, W = self.img_hw[k]
            if im.shape[:2] != (H, W):
                im = cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA)
            out[k] = np.moveaxis(im, -1, 0).astype(np.float32) / 255.0
        q = np.array(obs["qpos"] if "qpos" in obs else obs[self.lowdim_key], dtype=np.float32).copy()
        assert q.shape == self.state_min.shape, (q.shape, self.state_min.shape)
        if self.grasp_state_from_command and self.binary_dims:
            q[self.binary_dims] = self.last_grasp          # data convention: grasp state == last command
        self._check_ood(q)
        out[self.lowdim_key] = q
        return out

    def _check_ood(self, q):
        rng = np.maximum(self.state_max - self.state_min, 1e-4)
        rel = np.where(q > self.state_max, (q - self.state_max) / rng, np.where(q < self.state_min, (self.state_min - q) / rng, 0.0))
        self.last_ood_dims = [int(d) for d in np.nonzero(rel > self.ood_margin)[0] if d not in self.binary_dims]
        for d in self.last_ood_dims:
            if d not in self._ood_warned:
                self._ood_warned.add(d)
                name = self.spec.get("state_names", [str(i) for i in range(len(q))])[d]
                warnings.warn(f"qpos dim {d} ({name}) = {q[d]:.3f} is outside the training range "
                              f"[{self.state_min[d]:.3f}, {self.state_max[d]:.3f}] by >{self.ood_margin * 100:.0f}% of the range")

    def reset(self):
        self.buffer.reset()
        self.last_pred = None
        self.last_grasp = np.zeros(len(self.binary_dims), dtype=np.float32)
        self._ood_warned = set()

    # ---- prediction -----------------------------------------------------------------------------
    @torch.no_grad()
    def predict(self, obs_stack: Dict[str, np.ndarray]) -> np.ndarray:
        """obs_stack: {key: (To, ...)} preprocessed. Returns (H, Da) raw actions (K samples reduced)."""
        K = self.n_samples
        obs_t = dict_apply(obs_stack, lambda x: torch.from_numpy(np.ascontiguousarray(x)).unsqueeze(0).to(self.device))
        if K > 1:
            obs_t = dict_apply(obs_t, lambda x: x.expand(K, *x.shape[1:]).contiguous())
        result = self.policy.predict_action(obs_t)
        pred = result["action_pred"].float().cpu().numpy()   # (K, H, Da)
        if K == 1:
            return pred[0]
        if self.sample_reduce == "mean":
            return pred.mean(0)
        if self.sample_reduce == "median":
            return np.median(pred, axis=0)
        if self.sample_reduce == "first":
            return pred[0]
        raise ValueError(self.sample_reduce)

    def predict_chunk(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """Push one observation and return the full raw chunk (H, Da); index To-1+k = frame offset k."""
        self.buffer.push(self.prep_obs(obs))
        self.last_pred = self.predict(self.buffer.stacked())
        return self.last_pred

    def apply_grasp(self, actions: np.ndarray, last_grasp: Optional[np.ndarray] = None) -> np.ndarray:
        """Threshold binary dims with hysteresis, sequentially along the chunk."""
        act = actions.copy()
        g = self.last_grasp.copy() if last_grasp is None else np.asarray(last_grasp, dtype=np.float32).copy()
        hi, lo = self.grasp_threshold + self.grasp_hysteresis / 2, self.grasp_threshold - self.grasp_hysteresis / 2
        for t in range(act.shape[0]):
            for j, d in enumerate(self.binary_dims):
                v = act[t, d]
                if g[j] < 0.5 and v > hi:
                    g[j] = 1.0
                elif g[j] > 0.5 and v < lo:
                    g[j] = 0.0
                act[t, d] = g[j]
        return act

    def step(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """Synchronous use: returns n_action_steps actions starting `n_latency_steps` frames after the
        observation (assumes the whole returned window is executed)."""
        pred = self.predict_chunk(obs)
        s = self.n_obs_steps - 1 + self.n_latency_steps
        act = self.apply_grasp(pred[s:s + self.n_action_steps])
        if self.binary_dims:
            self.last_grasp = act[-1, self.binary_dims].copy()
        return act

    def set_last_grasp(self, grasp_cmd):
        self.last_grasp = np.asarray(grasp_cmd, dtype=np.float32).reshape(-1).copy()

    def contract(self) -> dict:
        names = self.spec.get("state_names")
        return dict(
            obs={**{k: dict(type="BGR uint8 HxWx3 (any size; resized INTER_AREA to trained_hw, center-cropped by the encoder)",
                            trained_hw=list(self.img_hw[k])) for k in self.rgb_keys},
                 "qpos": dict(type="float32", dim=len(self.state_min), names=names,
                              grasp_dims="filled with the LAST GRASP COMMAND by the wrapper (data convention), measured hand state is ignored",
                              training_range=dict(min=self.state_min.round(3).tolist(), max=self.state_max.round(3).tolist()),
                              ood_check=f"warn if a dim leaves the training range by >{self.ood_margin * 100:.0f}% of its range")},
            action=dict(dim=self.action_dim, names=self.spec.get("action_names"), units="absolute joint targets (rad); binary dims in {0,1}",
                        binary_dims=self.binary_dims, grasp_threshold=self.grasp_threshold, grasp_hysteresis=self.grasp_hysteresis,
                        tracking="the data was recorded with action[t] ~= qpos[t+2..3] at 30 Hz; the joint controller should track similarly"),
            timing=dict(fps=self.fps, n_obs_steps=self.n_obs_steps, horizon=self.horizon, n_latency_steps=self.n_latency_steps,
                        n_action_steps=self.n_action_steps, chunk_index_rule="frame offset k from the observation frame -> pred[n_obs_steps-1+k]",
                        sampler=self.sampler_desc, n_samples=self.n_samples, sample_reduce=self.sample_reduce),
        )


class ChunkScheduler:
    """Frame-indexed chunk bookkeeping for a 30 Hz control loop (no threads, no clock).

    add_chunk(pred, f_obs): register a prediction made from the observation at frame f_obs.
    command(f): action for frame f from the newest chunk that covers it, blended with the previous
    chunk over the first `blend_steps` frames after a switch; grasp dims thresholded with hysteresis.
    """

    def __init__(self, policy: G1DPPolicy, blend_steps: int = 4, max_offset: Optional[int] = None):
        self.p = policy
        self.To = policy.n_obs_steps
        self.H = policy.horizon
        self.blend_steps = int(blend_steps)
        self.max_offset = (self.H - self.To) if max_offset is None else int(max_offset)
        self.reset()

    def reset(self):
        self.chunks: List[tuple] = []     # (f_obs, pred)
        self.switch_frame = None
        self.prev_chunk = None
        self.grasp = self.p.last_grasp.copy()
        self.n_switches = 0
        self.n_starved = 0

    def add_chunk(self, pred: np.ndarray, f_obs: int):
        if self.chunks:
            self.prev_chunk = self.chunks[-1]
        self.chunks = [(f_obs, pred)]
        self.switch_frame = None  # set on first use
        self.n_switches += 1

    def _lookup(self, chunk, f):
        f_obs, pred = chunk
        k = f - f_obs
        if k < 0 or k > self.max_offset:
            return None
        return pred[self.To - 1 + k]

    def has_chunk(self):
        return len(self.chunks) > 0

    def command(self, f: int) -> Optional[np.ndarray]:
        if not self.chunks:
            self.n_starved += 1
            return None
        a = self._lookup(self.chunks[-1], f)
        if a is None:
            self.n_starved += 1
            return None
        a = a.copy()
        if self.switch_frame is None:
            self.switch_frame = f
        if self.prev_chunk is not None and self.blend_steps > 0:
            b = self._lookup(self.prev_chunk, f)
            i = f - self.switch_frame
            if b is not None and i < self.blend_steps:
                w = (i + 1) / (self.blend_steps + 1)          # weight of the new chunk ramps 1/(n+1) .. n/(n+1)
                a[:] = w * a + (1.0 - w) * b
        # grasp hysteresis on the executed stream
        out = self.p.apply_grasp(a[None], last_grasp=self.grasp)[0]
        if self.p.binary_dims:
            self.grasp = out[self.p.binary_dims].copy()
            self.p.last_grasp = self.grasp.copy()             # fed back into the next observation
        return out


class AsyncPolicy:
    """Real-time wrapper: inference in a background thread, one in flight at a time.

        ap = AsyncPolicy(policy, replan_every=policy.n_action_steps, blend_steps=4)
        ap.reset()
        for f in range(...):                     # once per frame at fps
            cmd = ap.tick(obs_f, f)              # None until the first chunk arrives -> hold position
            if cmd is not None: send(cmd)
    `tick` pushes the observation into the history, launches a new inference when the previous one
    has finished and `replan_every` frames have passed since the last launch, and returns the
    command for frame f from the scheduler.
    """

    def __init__(self, policy: G1DPPolicy, replan_every: Optional[int] = None, blend_steps: int = 4):
        self.p = policy
        self.sched = ChunkScheduler(policy, blend_steps=blend_steps)
        self.replan_every = int(policy.n_action_steps if replan_every is None else replan_every)
        self._lock = threading.Lock()
        self._thread = None
        self._result = None
        self._error = None
        self.last_launch = None
        self.latencies: List[float] = []

    def reset(self):
        self.join()
        self.p.reset()
        self.sched.reset()
        self._result = None
        self._error = None
        self.last_launch = None
        self.latencies.clear()

    def join(self):
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def _run(self, obs_stack, f_obs):
        t0 = time.time()
        try:
            pred = self.p.predict(obs_stack)
            with self._lock:
                self._result = (f_obs, pred, time.time() - t0)
        except Exception as exc:
            with self._lock:
                self._error = exc

    def busy(self):
        return self._thread is not None and self._thread.is_alive()

    def tick(self, obs: Dict[str, np.ndarray], f: int) -> Optional[np.ndarray]:
        self.p.buffer.push(self.p.prep_obs(obs))
        with self._lock:
            if self._error is not None:
                raise RuntimeError("Background policy inference failed") from self._error
            res, self._result = self._result, None
        if res is not None:
            f_obs, pred, lat = res
            self.latencies.append(lat)
            self.sched.add_chunk(pred, f_obs)
        if not self.busy() and (self.last_launch is None or f - self.last_launch >= self.replan_every or not self.sched.has_chunk()):
            self.join()
            self.last_launch = f
            self._thread = threading.Thread(target=self._run, args=(self.p.buffer.stacked(), f), daemon=True)
            self._thread.start()
        return self.sched.command(f)
