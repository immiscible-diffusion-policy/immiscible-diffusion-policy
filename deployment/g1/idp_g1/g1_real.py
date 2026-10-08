"""Real Unitree G1 backend for the idp_g1 deploy stack.

Implements `RobotInterface` on top of the same robot stack that collected the banana data
(yuxin's unified-finetuning-manager / xr_teleoperate):

  * arms   — `G1_29_ArmController` from UFM's vendored `openpi_g1_control` (DDS `rt/arm_sdk` in
             motion mode / `rt/lowcmd` in debug mode; internal 250 Hz PD thread with velocity
             clipping, we only set 14-dim absolute targets at the policy rate). Simple analytic
             gravity compensation (`use_gravity_compensation=True` on every command), matching
             `ufm/execution/live_g1.py`.
  * hands  — `Dex3DirectController` (topics `rt/dex3/{left,right}/cmd`); the policy's binary grasp
             dims are mapped to the same closed-hand joint presets UFM uses
             (`ufm/runtime/openpi_g1_utils.binary_gripper_to_hand_joints`).
  * images — ZMQ SUB (CONFLATE) on the robot's image server (default ROBOT_IP:5555), which
             publishes JPEG( hconcat[ ego 480x640 | wide 480x640 ] ); split at `head_width`.

**Channel order: the panels are passed on as decoded by OpenCV, i.e. BGR, on purpose.** The
HDF5 recordings hold the image-server frames verbatim (BGR — see g1_wam/wam/hdf5_image_server.py
and the blue-looking banana), so training consumed BGR panels and deployment must feed the same.

qpos(16) = [arm q(14) from DDS | last grasp COMMANDS(2)]  — the recorded "grasp state" is a
command echo, and `G1DPPolicy` overwrites those dims with its own last command anyway.

Usage with the existing staged tools (all of them accept `--robot module:Class`):

    --robot idp_g1.g1_real:UnitreeG1Robot \
    --robot-kwargs '{"dds_interface": "<nic>", "image_server_address": "ROBOT_IP"}'

DDS note: initialise once per process. `dds_interface` must be the NIC wired to the robot
(e.g. "enp3s0"); None lets cyclonedds pick.
"""
from __future__ import annotations

import atexit
import json
import importlib.util
import os
import signal
import sys
import threading
import time
from typing import Dict, Optional

import numpy as np

from .robot_interface import RobotInterface

# ---------------------------------------------------------------------------------------------
# vendored robot stack (yuxin's repos). Loaded file-by-file so we do not pull in the package
# __init__, which imports the casadi/pinocchio IK that this env does not have.
# Set this to the experiment stack's openpi_g1_control/robot_control directory.
# unitree_sdk2py must be installed independently.
_VENDOR_CTRL = os.environ.get("G1_CONTROLLER_DIR", "")


def _load_vendor_module(name: str, path: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# cyclonedds config used by unitree_sdk2py when an interface name is given. The vendored template
# relies on MULTICAST discovery only; over the USB NIC our participant announcement often does not
# reach the robot, so discovery completes only when the robot's own periodic announcement comes
# round (cyclonedds default SPDP interval is 30 s) -> the "[G1_29] Waiting for DDS" bar sits there
# for up to half a minute on every connect. Adding the robot as a unicast Peer makes our
# announcement go straight to it and it answers immediately. Disable with G1DP_DDS_STOCK=1;
# change the robot address with G1DP_DDS_PEER.
_DDS_PEER = os.environ.get("G1DP_DDS_PEER", "")
_DDS_CONFIG_WITH_PEER = """<?xml version="1.0" encoding="UTF-8" ?>
    <CycloneDDS>
        <Domain Id="any">
            <General>
                <Interfaces>
                    <NetworkInterface name="$__IF_NAME__$" priority="default" multicast="default"/>
                </Interfaces>
            </General>
            <Discovery>
                <Peers>
                    <Peer address="%s"/>
                </Peers>
                <ParticipantIndex>auto</ParticipantIndex>
                <MaxAutoParticipantIndex>64</MaxAutoParticipantIndex>
            </Discovery>
        </Domain>
    </CycloneDDS>""" % _DDS_PEER


def _import_robot_stack():
    if not _VENDOR_CTRL:
        raise RuntimeError("Set G1_CONTROLLER_DIR to the compatible robot_control directory (see README)")
    for filename in ("robot_arm.py", "dex3_hand.py"):
        if not os.path.isfile(os.path.join(_VENDOR_CTRL, filename)):
            raise FileNotFoundError(os.path.join(_VENDOR_CTRL, filename))
    arm_mod = _load_vendor_module("g1dp_vendor_robot_arm", os.path.join(_VENDOR_CTRL, "robot_arm.py"))
    hand_mod = _load_vendor_module("g1dp_vendor_dex3_hand", os.path.join(_VENDOR_CTRL, "dex3_hand.py"))
    from unitree_sdk2py.core import channel as _channel
    from unitree_sdk2py.core import channel_config as _channel_config
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize  # noqa: F401
    if _DDS_PEER and not os.environ.get("G1DP_DDS_STOCK"):
        for m in (_channel, _channel_config):
            if hasattr(m, "ChannelConfigHasInterface"):
                m.ChannelConfigHasInterface = _DDS_CONFIG_WITH_PEER
    return arm_mod, hand_mod, ChannelFactoryInitialize


# Where each process leaves the last arm target it commanded, so the next one can resume EXACTLY that
# hold instead of re-seeding from the measured pose. Seeding from the measured pose costs one gravity
# sag per process start (target = measured -> zero PD error -> the arm settles a few cm lower), which
# accumulates over a session of goto_init -> replay -> openloop -> runner.
_LAST_TARGET_FILE = os.path.join(os.path.expanduser("~/.cache"), "idp_g1_last_arm_target.json")
_LAST_TARGET_MAX_AGE_S = 3600.0
_LAST_TARGET_MAX_DIST_RAD = 0.25      # measured must be this close to the saved target, else ignore it


def _save_last_target(q14):
    try:
        os.makedirs(os.path.dirname(_LAST_TARGET_FILE), exist_ok=True)
        with open(_LAST_TARGET_FILE, "w") as f:
            json.dump(dict(t=time.time(), q=[float(v) for v in np.asarray(q14).reshape(14)]), f)
    except Exception:
        pass


def _load_last_target():
    try:
        d = json.load(open(_LAST_TARGET_FILE))
        if time.time() - float(d["t"]) > _LAST_TARGET_MAX_AGE_S:
            return None
        return np.asarray(d["q"], dtype=np.float32).reshape(14)
    except Exception:
        return None


def _clear_last_target():
    try:
        os.remove(_LAST_TARGET_FILE)
    except Exception:
        pass


_REMOTE_POSE_FILE = os.path.expanduser(os.environ.get("G1_REMOTE_POSE", ""))


def _load_remote_pose_arm_q():
    """14 arm angles the internal controller holds after the remote sequence (see `idp_g1.capture_pose`), or None."""
    try:
        q = np.asarray(json.load(open(_REMOTE_POSE_FILE))["arm_q"], dtype=np.float32).reshape(14)
        return q if np.isfinite(q).all() else None
    except Exception:
        return None


def _hold_on_start_controller(arm_mod, seed_q=None):
    """Subclass of the vendored G1_29_ArmController whose publish thread starts from the MEASURED pose.

    The vendored __init__ sets `q_target = zeros(14)` and starts the 250 Hz publish thread, so the
    first thing any process does after connecting is drive both arms to q=0 at the 40 rad/s velocity
    clip — i.e. slam them down. Nobody noticed while every session started with the arms already at
    zero; with the no-hand-off workflow (goto_init --hold -> runner) the arms are up when the next
    process connects. Seeding the target with the current joint angles (+ gravity compensation) makes
    connect() a no-op for the arms: they hold where they are until the first command.
    """
    Base = arm_mod.G1_29_ArmController

    class HoldOnStartArmController(Base):
        def _ctrl_motor_state(self):
            q = np.asarray(self.get_current_dual_arm_q(), dtype=np.float32)
            if seed_q is not None:
                d = float(np.abs(seed_q - q).max())
                if d <= _LAST_TARGET_MAX_DIST_RAD:
                    print(f"[g1_real] resuming the previous process's arm target (measured is {d:.3f} rad from it)",
                          flush=True)
                    q = np.asarray(seed_q, dtype=np.float32)
                else:
                    print(f"[g1_real] previous arm target is {d:.3f} rad from the measured pose — ignoring it, "
                          f"holding the measured pose", flush=True)
            with self.ctrl_lock:
                self.q_target = q
                self.tauff_target = np.clip(self.compute_gravity_compensation(q),
                                            -self.torque_limits, self.torque_limits).astype(np.float32)
            return super()._ctrl_motor_state()

    HoldOnStartArmController.__name__ = Base.__name__ + "HoldOnStart"
    return HoldOnStartArmController


# Binary grasp -> Dex3 7-DOF joint targets (DDS motor order), copied verbatim from
# ufm/runtime/openpi_g1_utils.py (magnitudes inside the URDF limits; left[i] == -right[i]).
_LEFT_HAND_CLOSED = np.array([0.0, 0.88, 1.68, -1.48, -1.68, -1.48, -1.68], dtype=np.float32)
_RIGHT_HAND_CLOSED = np.array([0.0, -0.88, -1.68, 1.48, 1.68, 1.48, 1.68], dtype=np.float32)
_HAND_OPEN = np.zeros(7, dtype=np.float32)


class _ImageFeed:
    """Latest-frame ZMQ subscriber for the G1 image server; splits [head|wide] composites."""

    def __init__(self, address: str, port: int, head_width: int = 640, timeout_ms: int = 1000):
        import cv2
        import zmq
        self.cv2 = cv2
        self._zmq = zmq
        self.head_width = int(head_width)
        self._ctx = zmq.Context()
        self._sock = self._ctx.socket(zmq.SUB)
        self._sock.setsockopt(zmq.CONFLATE, 1)
        self._sock.setsockopt_string(zmq.SUBSCRIBE, "")
        self._sock.setsockopt(zmq.RCVTIMEO, int(timeout_ms))
        self._sock.connect(f"tcp://{address}:{port}")
        self._last: Optional[np.ndarray] = None

    def read(self) -> Dict[str, np.ndarray]:
        try:
            payload = self._sock.recv()
            frame = self.cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), self.cv2.IMREAD_COLOR)
            if frame is None:
                raise RuntimeError("image server frame failed to decode")
            self._last = frame
        except self._zmq.Again:
            raise TimeoutError("No fresh frame from the image server")
        h, w = frame.shape[:2]
        hw = self.head_width
        if hw <= 0 or w <= hw:
            raise ValueError("Expected image-server composite [ego | wide]")
        ego = frame[:, :hw]
        wide = frame[:, hw:]
        if wide.shape[1] != ego.shape[1]:
            wide = self.cv2.resize(wide, (ego.shape[1], ego.shape[0]), interpolation=self.cv2.INTER_AREA)
        # BGR on purpose — matches what the recorder stored and the policy trained on (see module doc).
        return {"ego_cam": np.ascontiguousarray(ego), "wide_cam": np.ascontiguousarray(wide)}

    def close(self):
        try:
            self._sock.close(0)
            self._ctx.term()
        except Exception:
            pass


class UnitreeG1Robot(RobotInterface):
    name = "unitree_g1"

    def __init__(self,
                 dds_interface: Optional[str] = None,
                 image_server_address: Optional[str] = None,
                 image_server_port: int = 5555,
                 head_width: int = 640,
                 motion_mode: bool = True,
                 enable_hands: bool = True,
                 gravity_compensation: bool = True,
                 auto_release: bool = True,
                 release_seconds: float = 4.0,
                 release_ramp_seconds: float = 2.0,
                 waist_upright: bool = True,
                 fps: float = 30.0):
        if not image_server_address:
            raise ValueError("Set image_server_address in the robot configuration")
        self.dds_interface = dds_interface
        # In arm-sdk mode our controller also pins waist pitch/roll (kp 350) at whatever angle they had
        # when the process connected, and the internal balance controller cannot correct the torso while
        # the sdk weight is 1. Connecting mid-recovery froze a 10 deg forward lean on 2026-09-11. With
        # `waist_upright` the waist targets are ramped to the saved remote-controller default pose
        # (G1_REMOTE_POSE from `idp_g1.capture_pose`) right after connecting.
        self.waist_upright = bool(waist_upright)
        # "Park" pose: where the arms go on exit when control is NOT handed back (auto_release=False).
        # The tools set it to the task's dataset initial pose, so every program ends where the next
        # one (and every recorded demo) starts. None -> just hold the current pose.
        self.park_q16: Optional[np.ndarray] = None
        self.park_seconds = 4.0
        # how the arms are handed back on exit: `release_seconds` of smoothstep lowering to q=0, then
        # `release_ramp_seconds` of arm-sdk weight ramp 1 -> 0. Defaults reproduce the vendored
        # behaviour (4 s + 2 s); goto_init uses slower values so the balance controller is not kicked.
        self.release_seconds = float(release_seconds)
        self.release_ramp_seconds = float(release_ramp_seconds)
        self.motion_mode = bool(motion_mode)
        self.enable_hands = bool(enable_hands)
        self.gravity_compensation = bool(gravity_compensation)
        self.auto_release = bool(auto_release)
        self._released = False
        self.fps = float(fps)
        self.arm = None
        self.hand = None
        self.feed = _ImageFeed(image_server_address, image_server_port, head_width)
        self._last_grasp = np.zeros(2, dtype=np.float32)
        self._t0 = time.time()
        self._lock = threading.Lock()

    # -- lifecycle ---------------------------------------------------------------------------
    def connect(self):
        if _load_remote_pose_arm_q() is None:
            raise RuntimeError("Set G1_REMOTE_POSE to your recorded controller pose; see idp_g1.capture_pose")
        arm_mod, hand_mod, ChannelFactoryInitialize = _import_robot_stack()
        if self.dds_interface:
            ChannelFactoryInitialize(0, self.dds_interface)
        else:
            ChannelFactoryInitialize(0)
        self.arm = _hold_on_start_controller(arm_mod, seed_q=_load_last_target())(
            motion_mode=self.motion_mode, control_waist=False,
            simulation_mode=False, dds_already_initialized=True,
            dds_interface=self.dds_interface)
        self.arm.speed_gradual_max()          # ramp the velocity limit over ~5 s at startup
        if self.waist_upright:
            self._waist_to_remote_pose(arm_mod)
        if self.enable_hands:
            self.hand = hand_mod.Dex3DirectController(
                dds_already_initialized=True, dds_interface=self.dds_interface)
        self._t0 = time.time()
        if self.auto_release:
            # any exit — normal return, `kill` (SIGTERM), Ctrl-C, uncaught exception — lowers the
            # arms and hands control back to the internal controller. `kill -9` cannot be caught:
            # process-kill recovery belongs to the external robot controller stack.
            atexit.register(self.release)
            for sig in (signal.SIGTERM, signal.SIGHUP):
                try:
                    signal.signal(sig, self._on_signal)
                except (ValueError, OSError):
                    pass                      # not the main thread / unsupported

    def _on_signal(self, signum, _frame):
        print(f"[g1_real] signal {signum}: lowering arms and releasing control ...", flush=True)
        sys.exit(0)                            # unwinds + runs the atexit release

    def _waist_to_remote_pose(self, arm_mod, seconds: float = 2.0, max_step_rad: float = 0.5):
        """Ramp waist yaw/roll/pitch targets from the measured angles to the saved remote-pose values."""
        try:
            path = _REMOTE_POSE_FILE
            doc = json.load(open(path))
            ref = np.asarray(doc["all_motor_q"], dtype=np.float32)
        except Exception:
            print("[g1_real] no saved remote pose (G1_REMOTE_POSE) — waist left as measured", flush=True)
            return
        JI = arm_mod.G1_29_JointIndex
        idx = [int(JI.kWaistYaw), int(JI.kWaistRoll), int(JI.kWaistPitch)]
        q_now = np.asarray(self.arm.get_current_motor_q(), dtype=np.float32)[idx]
        q_ref = ref[idx]
        d = q_ref - q_now
        if float(np.abs(d).max()) > max_step_rad:
            print(f"[g1_real] waist is {np.abs(d).max():.2f} rad from the saved remote pose — too far, left as is", flush=True)
            return
        if float(np.abs(d).max()) < 0.01:
            return
        print(f"[g1_real] waist yaw/roll/pitch {np.round(q_now, 3).tolist()} -> remote pose "
              f"{np.round(q_ref, 3).tolist()} over {seconds:.1f} s", flush=True)
        steps = max(1, int(seconds * 50))
        for i in range(steps):
            t = (i + 1) / steps
            t = t * t * (3.0 - 2.0 * t)
            q = q_now + t * d
            with self.arm.ctrl_lock:
                self.arm.waist_yaw_target = float(q[0])          # the publish loop writes yaw every tick
                self.arm.msg.motor_cmd[JI.kWaistRoll].q = float(q[1])   # roll/pitch are only set at init
                self.arm.msg.motor_cmd[JI.kWaistPitch].q = float(q[2])
            time.sleep(1.0 / 50)

    def release(self, seconds: Optional[float] = None, ramp_seconds: Optional[float] = None):
        """Lower the arms to zero, open the hands, ramp the arm-sdk weight to 0 (idempotent).

        Slow on purpose: the arms are ~2 kg each on a ~0.5 m lever, and in motion mode the balance
        controller sees every arm move as a CoM disturbance — a fast drop or an abrupt weight hand-off
        makes the robot step backwards. Both durations default to the constructor values.

        The arms are lowered to the INTERNAL CONTROLLER'S OWN default pose (G1_REMOTE_POSE,
        elbows bent ~1 rad) when that file exists, and release refuses to continue if the saved pose is unavailable. Handing off from q=0 made the
        internal controller swing both elbows by ~1 rad during the weight ramp — that CoM jerk is what
        stepped the robot backwards on 2026-09-11; handing off *at* its pose moves nothing.
        """
        if self._released or self.arm is None:
            return
        self._released = True
        seconds = self.release_seconds if seconds is None else float(seconds)
        ramp_seconds = self.release_ramp_seconds if ramp_seconds is None else float(ramp_seconds)
        try:
            if self.hand is not None:
                self.hand.ctrl_dual_hand(np.zeros(7, np.float32), np.zeros(7, np.float32))
            q0 = np.asarray(self.arm.get_current_dual_arm_q(), dtype=np.float32)
            q_target = _load_remote_pose_arm_q()
            where = "the internal controller's default pose" if q_target is not None else "q=0 (no G1_REMOTE_POSE)"
            if q_target is None:
                raise RuntimeError("Saved controller pose is unavailable; refusing an uncalibrated hand-off")
            steps = max(1, int(seconds * self.fps))
            print(f"[g1_real] moving arms to {where} over {seconds:.1f} s ...", flush=True)
            for i in range(steps):
                t = (i + 1) / steps
                t = t * t * (3.0 - 2.0 * t)
                self.arm.ctrl_dual_arm(q0 + t * (q_target - q0), np.zeros(14, np.float32),
                                       use_gravity_compensation=self.gravity_compensation)
                time.sleep(1.0 / self.fps)
            time.sleep(0.5)
            # hand the arms back to the internal controller at our own pace. Same mechanism as the
            # vendored ramp (kNotUsedJoint0.q is the arm-sdk blend weight). The vendored go_home()
            # is NOT used: it would drive the arms to q=0 first.
            arm_mod = sys.modules.get("g1dp_vendor_robot_arm")
            weight_idx = arm_mod.G1_29_JointIndex.kNotUsedJoint0
            n = max(2, int(ramp_seconds / 0.02))
            print(f"[g1_real] handing arm control back over {ramp_seconds:.1f} s ...", flush=True)
            for w in np.linspace(1.0, 0.0, num=n):
                self.arm.msg.motor_cmd[weight_idx].q = float(w)
                time.sleep(0.02)
            print("[g1_real] arms down, hands open, arm-sdk control released.", flush=True)
        except Exception as exc:
            print(f"[g1_real] release failed ({type(exc).__name__}: {exc}) — run "
                  f"your existing controller hand-off tool manually.", flush=True)

    def close(self):
        if self._released:
            pass
        elif self.auto_release:
            self.release()
        elif self.park_q16 is not None and self.arm is not None:
            try:
                q_now = np.asarray(self.arm.get_current_dual_arm_q(), dtype=np.float32)
                d = float(np.abs(q_now - np.asarray(self.park_q16, dtype=np.float32)[:14]).max())
                print(f"[g1_real] parking at the initial pose over {self.park_seconds:.1f} s ({d:.2f} rad away); "
                      f"arm-sdk keeps holding it", flush=True)
                self.home(self.park_q16, duration=self.park_seconds)
            except Exception as exc:
                print(f"[g1_real] park failed ({type(exc).__name__}: {exc}); holding the current pose", flush=True)
                self.stop()
        else:
            self.stop()
        if self._released:
            _clear_last_target()                       # control handed back: nothing to resume
        elif getattr(self, "_last_arm_cmd", None) is not None:
            _save_last_target(self._last_arm_cmd)      # arms stay held at this target after we exit
        if self.hand is not None:
            try:
                self.hand.stop()
            except Exception:
                pass
            self.hand = None
        if self.arm is not None:
            stopper = getattr(self.arm, "stop", None)
            if callable(stopper):
                try:
                    stopper()
                except Exception:
                    pass
            self.arm = None
        self.feed.close()

    # -- RobotInterface ----------------------------------------------------------------------
    def get_obs(self) -> Dict[str, np.ndarray]:
        assert self.arm is not None, "call connect() first"
        imgs = self.feed.read()
        q14 = np.asarray(self.arm.get_current_dual_arm_q(), dtype=np.float32)
        qpos = np.concatenate([q14, self._last_grasp]).astype(np.float32)
        return {**imgs, "qpos": qpos, "t": time.time() - self._t0}

    def send_action(self, action: np.ndarray):
        assert self.arm is not None, "call connect() first"
        a = np.asarray(action, dtype=np.float32).reshape(16)
        if not np.isfinite(a).all():
            raise ValueError("Non-finite robot action")
        with self._lock:
            self.arm.ctrl_dual_arm(a[:14], np.zeros(14, dtype=np.float32),
                                   use_gravity_compensation=self.gravity_compensation)
            if self.hand is not None:
                left = _LEFT_HAND_CLOSED if a[14] > 0.5 else _HAND_OPEN
                right = _RIGHT_HAND_CLOSED if a[15] > 0.5 else _HAND_OPEN
                self.hand.ctrl_dual_hand(left, right)
            self._last_grasp = (a[14:16] > 0.5).astype(np.float32)
            self._last_arm_cmd = a[:14].copy()

    def home(self, q_target: Optional[np.ndarray] = None, duration: float = 3.0):
        """Smooth cosine move of the ARMS to `q_target[:14]` (grasp dims -> open unless commanded)."""
        assert self.arm is not None, "call connect() first"
        if q_target is None:
            q_target = np.zeros(16, dtype=np.float32)
        q_target = np.asarray(q_target, dtype=np.float32).reshape(-1)[:16]
        q_now = np.asarray(self.arm.get_current_dual_arm_q(), dtype=np.float32)
        steps = max(1, int(round(duration * self.fps)))
        for i in range(steps):
            t = (i + 1) / steps
            t = t * t * (3.0 - 2.0 * t)                       # smoothstep, as in ufm live_g1
            interp = q_now + t * (q_target[:14] - q_now)
            self.send_action(np.concatenate([interp, q_target[14:16]]))
            time.sleep(1.0 / self.fps)

    def stop(self):
        """Hold the pose (grasp state unchanged).

        Holds the LAST COMMANDED arm target when the arm is tracking it (within 0.25 rad), and the
        measured pose otherwise. Holding the measured pose would re-anchor the PD target on the
        gravity-sagged position and let the arm drop a few cm more at every stop / process exit.
        """
        if self.arm is None:
            return
        q_now = np.asarray(self.arm.get_current_dual_arm_q(), dtype=np.float32)
        last = getattr(self, "_last_arm_cmd", None)
        hold = last if (last is not None and float(np.abs(last - q_now).max()) <= _LAST_TARGET_MAX_DIST_RAD) else q_now
        self.send_action(np.concatenate([hold, self._last_grasp]))
