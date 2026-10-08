"""Pure-logic tests for the deployment layer (no checkpoint / GPU needed).

    python -m unittest discover -s deployment/g1/tests -v
"""
import sys
import time
import types

import numpy as np

from idp_g1.infer import AsyncPolicy, ChunkScheduler, G1DPPolicy, ObsBuffer


class FakePolicy:
    """Minimal stand-in exposing what ChunkScheduler / AsyncPolicy use."""

    def __init__(self, To=2, H=32, Na=12, L=10, Da=16, binary=(14, 15), delay=0.0):
        self.n_obs_steps, self.horizon, self.n_action_steps, self.n_latency_steps = To, H, Na, L
        self.action_dim = Da
        self.binary_dims = list(binary)
        self.last_grasp = np.zeros(len(binary), dtype=np.float32)
        self.grasp_threshold, self.grasp_hysteresis = 0.5, 0.1
        self.buffer = ObsBuffer(To)
        self.delay = delay
        self.calls = 0

    # reuse the real implementations
    apply_grasp = G1DPPolicy.apply_grasp
    def reset(self):
        self.buffer.reset(); self.last_grasp[:] = 0
    def prep_obs(self, obs):
        return {"agent_pos": np.asarray(obs["qpos"], dtype=np.float32)}
    def predict(self, obs_stack):
        """Chunk whose arm value encodes the obs frame id: pred[To-1+k, 0] = f_obs + k."""
        self.calls += 1
        f_obs = float(obs_stack["agent_pos"][-1, 0])
        pred = np.zeros((self.horizon, self.action_dim), dtype=np.float32)
        pred[:, 0] = f_obs + (np.arange(self.horizon) - (self.n_obs_steps - 1))
        pred[:, 14] = 0.9 if f_obs >= 20 else 0.1          # grasp closes for chunks predicted at/after frame 20
        if self.delay:
            time.sleep(self.delay)
        return pred


def test_obs_buffer_padding():
    b = ObsBuffer(3); b.push({"x": np.array([1.0])}); s = b.stacked()
    assert s["x"].shape == (3, 1) and (s["x"] == 1.0).all()
    b.push({"x": np.array([2.0])}); assert b.stacked()["x"][:, 0].tolist() == [1.0, 1.0, 2.0]


def test_grasp_hysteresis():
    p = FakePolicy()
    a = np.zeros((6, 16), dtype=np.float32); a[:, 14] = [0.52, 0.56, 0.48, 0.46, 0.44, 0.3]
    out = G1DPPolicy.apply_grasp(p, a)
    # open->close needs > 0.55; close->open needs < 0.45 (0.48 / 0.46 stay closed, 0.44 opens)
    assert out[:, 14].tolist() == [0.0, 1.0, 1.0, 1.0, 0.0, 0.0], out[:, 14].tolist()


def test_scheduler_frame_offsets_and_blend():
    p = FakePolicy(); s = ChunkScheduler(p, blend_steps=2)
    assert s.command(0) is None and s.n_starved == 1
    pred0 = p.predict({"agent_pos": np.array([[0.0] * 16, [0.0] * 16])})
    s.add_chunk(pred0, f_obs=0)
    # action for frame f from a chunk anchored at f_obs=0 is pred[To-1+f] -> value f
    for f in (10, 11, 12):
        assert s.command(f)[0] == f
    # a later chunk anchored at f_obs=12 delivered at frame 22: values continue to be == frame (consistent),
    # blending two consistent chunks must not change the value
    pred1 = p.predict({"agent_pos": np.array([[12.0] * 16, [12.0] * 16])})
    s.add_chunk(pred1, f_obs=12)
    for f in (22, 23, 24):
        assert abs(s.command(f)[0] - f) < 1e-6
    # inconsistent chunk: new says value+1 -> blended 1/3, 2/3, then full
    pred2 = pred1.copy(); pred2[:, 0] += 1.0
    s.add_chunk(pred2, f_obs=12)
    v = [s.command(f)[0] - f for f in (25, 26, 27)]
    assert np.allclose(v, [1 / 3, 2 / 3, 1.0]), v
    # beyond usable offset -> starved
    assert s.command(12 + (p.horizon - p.n_obs_steps) + 1) is None


def test_scheduler_grasp_feedback():
    p = FakePolicy(); s = ChunkScheduler(p, blend_steps=0)
    s.add_chunk(p.predict({"agent_pos": np.array([[30.0] * 16] * 2)}), f_obs=30)
    a = s.command(40)
    assert a[14] == 1.0 and p.last_grasp[0] == 1.0      # closed, and fed back as the next qpos grasp state


def test_async_policy_timing():
    p = FakePolicy(delay=0.03); ap = AsyncPolicy(p, replan_every=12, blend_steps=0); ap.reset()
    cmds = []
    for f in range(60):
        cmds.append(ap.tick({"qpos": np.full(16, float(f))}, f))
        time.sleep(0.005)
    ap.join()
    assert cmds[0] is None                                   # first chunk not ready at frame 0
    got = [(f, c[0]) for f, c in enumerate(cmds) if c is not None]
    assert len(got) > 40 and all(abs(v - f) < 1e-6 for f, v in got), got[:5]   # every command is for the right frame
    assert 3 <= p.calls <= 6, p.calls                        # ~ one launch per replan_every frames
    assert len(ap.latencies) == p.calls - (1 if ap.busy() else 0) or len(ap.latencies) >= p.calls - 1



import unittest

class OriginalInferenceTests(unittest.TestCase):
    pass

for name, fn in list(globals().items()):
    if name.startswith("test_") and callable(fn):
        setattr(OriginalInferenceTests, name, lambda self, fn=fn: fn())

del name, fn
