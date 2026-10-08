"""Rate control and recording helpers adapted from the experiment deployment."""
from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional

import cv2
import numpy as np



class RateTimer:
    """Fixed-rate loop timer; records tick jitter and missed ticks."""

    def __init__(self, fps: float, realtime: bool = True):
        self.dt = 1.0 / fps
        self.realtime = realtime
        self.t0 = None
        self.k = 0
        self.late = []

    def wait(self):
        if not self.realtime:
            self.k += 1
            return
        now = time.perf_counter()
        if self.t0 is None:
            self.t0 = now
        target = self.t0 + self.k * self.dt
        if now < target:
            time.sleep(target - now)
        else:
            self.late.append(now - target)
        self.k += 1

    def stats(self):
        return dict(ticks=self.k, late_ticks=len(self.late), max_late_ms=float(max(self.late) * 1000) if self.late else 0.0,
                    mean_late_ms=float(np.mean(self.late) * 1000) if self.late else 0.0)


class Recorder:
    """Stores obs/cmd streams and writes an overlay mp4 of the camera views."""

    def __init__(self, out_dir: str, rgb_keys: List[str], fps: float, video: bool = True):
        self.out_dir = out_dir; os.makedirs(out_dir, exist_ok=True)
        self.rgb_keys = rgb_keys
        self.fps = fps
        self.q, self.cmd, self.t, self.txt = [], [], [], []
        self.vw = None
        self.video = video

    def add(self, obs: Dict[str, np.ndarray], cmd: Optional[np.ndarray], text: str = ""):
        self.q.append(np.asarray(obs["qpos"], dtype=np.float32).copy())
        self.cmd.append(None if cmd is None else np.asarray(cmd, dtype=np.float32).copy())
        self.t.append(float(obs.get("t", len(self.t) / self.fps)))
        if self.video:
            row = np.concatenate([np.asarray(obs[k]) for k in self.rgb_keys], axis=1).copy()
            if self.vw is None:
                self.vw = cv2.VideoWriter(os.path.join(self.out_dir, "cams.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                          int(round(self.fps)), (row.shape[1], row.shape[0]))
            cv2.putText(row, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            self.vw.write(row)

    def close(self, summary: dict):
        if self.vw is not None:
            self.vw.release()
        cmd = np.stack([c if c is not None else np.full(16, np.nan, np.float32) for c in self.cmd]) if self.cmd else np.zeros((0, 16))
        np.savez_compressed(os.path.join(self.out_dir, "log.npz"), qpos=np.stack(self.q) if self.q else np.zeros((0, 16)),
                            cmd=cmd, t=np.array(self.t))
        with open(os.path.join(self.out_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=1, default=float)
