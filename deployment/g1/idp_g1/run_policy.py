"""Run the paper-task image DP with an optional G1 interface or a synthetic robot.

The episode loop is adapted from g1_dp/deploy/episode_loop.py. Sampling uses
one ordinary policy sample, without hand selection or sample averaging.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import select
import sys
import termios
import time
import tty

import numpy as np

from .common import RateTimer, Recorder
from .infer import AsyncPolicy, G1DPPolicy
from .robot_interface import MockRobot, SafetyLimiter, urdf_arm_limits


class Keys:
    """Nonblocking terminal input, restoring terminal settings on exit."""

    def __init__(self, enabled):
        self.enabled = enabled
        self.old = None

    def __enter__(self):
        if self.enabled:
            self.fd = sys.stdin.fileno()
            self.old = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
        return self

    def __exit__(self, *exc):
        if self.old is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)

    def poll(self):
        if self.enabled and select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1).lower()
        return None


def run_session(robot, policy, config, limits, output, seconds, mock=False,
                video=True, init_seconds=3.0):
    """Shared loop; all robot commands pass through the configured interface."""
    init_pose = np.asarray(config["init_pose"], dtype=np.float32)
    if init_pose.shape != (16,) or not np.isfinite(init_pose).all():
        raise ValueError("init_pose must contain 16 finite values")
    limiter = SafetyLimiter(limits, max_step_rad=float(config["max_step_rad"]))
    limiter.reset(init_pose)  # Validate the task pose before connecting to hardware.
    async_policy = AsyncPolicy(policy, blend_steps=int(config["blend_steps"]))
    robot.fps = policy.fps
    if hasattr(robot, "park_q16"):
        robot.park_q16 = init_pose.copy()
        robot.park_seconds = init_seconds
    output.mkdir(parents=True, exist_ok=False)
    print(f"Session output: {output}", flush=True)
    recorder = None
    episode = 0
    frame = 0
    active = False
    initialized = False

    def start():
        nonlocal recorder, episode, frame, active, timer, started
        async_policy.reset()
        measured = robot.get_obs()["qpos"]
        limiter.reset(measured)
        # Both the task initial pose and grasp state are open at episode start.
        episode += 1
        frame = 0
        recorder = Recorder(str(output / f"ep{episode:03d}"), policy.rgb_keys,
                            policy.fps, video=video)
        timer = RateTimer(policy.fps)
        started = time.perf_counter()
        active = True
        print(f"Episode {episode}: running", flush=True)

    def stop(reason):
        nonlocal active, recorder
        active = False
        robot.stop()
        async_policy.join()
        if recorder is not None:
            recorder.close(dict(reason=reason, frames=frame, fps=policy.fps,
                                sampler=policy.sampler_desc, n_samples=1,
                                blend_steps=int(config["blend_steps"]),
                                starved_frames=async_policy.sched.n_starved,
                                inference_seconds=async_policy.latencies,
                                limiter=limiter.stats()))
            recorder = None
        print(f"Episode stopped ({reason}); robot holds its target", flush=True)

    timer = None
    started = None
    try:
        robot.connect()
        with Keys(enabled=not mock) as keys:
            if mock:
                robot.home(init_pose, duration=0)
                initialized = True
                start()
            else:
                print("i: initial pose | r: run | s: hold | q: quit", flush=True)
            while True:
                key = keys.poll()
                if key == "q" or (output / "STOP").exists():
                    if active:
                        stop("quit_or_stop_file")
                    break
                if active:
                    if key == "s":
                        stop("key_s")
                        initialized = False
                        continue
                    if time.perf_counter() - started >= seconds:
                        stop("time_cap")
                        initialized = False
                        if mock:
                            break
                        continue
                    observation = robot.get_obs()
                    command = async_policy.tick(observation, frame)
                    if command is not None:
                        command = limiter(command)
                        robot.send_action(command)
                    recorder.add(observation, command, text=f"ep{episode} f{frame}")
                    frame += 1
                    timer.wait()
                elif key == "i":
                    robot.home(init_pose, duration=init_seconds)
                    initialized = True
                    print("At initial pose; position the object, then press r", flush=True)
                elif key == "r":
                    if initialized:
                        start()
                    else:
                        print("Press i to return to the initial pose first", flush=True)
                else:
                    time.sleep(0.02)
    except KeyboardInterrupt:
        if active:
            stop("keyboard_interrupt")
    except Exception:
        if active:
            stop("error")
        raise
    finally:
        try:
            async_policy.join()
        finally:
            robot.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--task-config", required=True,
                        help="configs/banana.json or configs/soda.json")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--mock", action="store_true", help="synthetic frames and joints; no hardware")
    mode.add_argument("--robot-config", help="local robot JSON with camera address and DDS interface")
    mode.add_argument("--check-checkpoint", action="store_true", help="load and predict without robot I/O")
    parser.add_argument("--urdf", help="G1 URDF for this hardware's arm joint limits")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seconds", type=float, default=30)
    parser.add_argument("--init-seconds", type=float, default=3)
    parser.add_argument("--out", default=None)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--release-on-exit", action="store_true",
                        help="return to the saved controller pose and hand control back; otherwise park at task pose")
    args = parser.parse_args()
    if not np.isfinite(args.seconds) or args.seconds <= 0 or not np.isfinite(args.init_seconds) or args.init_seconds <= 0:
        parser.error("durations must be finite and positive")
    if args.robot_config and (not args.urdf or not sys.stdin.isatty()):
        parser.error("real deployment requires --urdf and an interactive terminal")
    with open(args.task_config) as stream:
        config = json.load(stream)
    if config.get("image_channel_order") != "BGR":
        parser.error("paper-task camera inputs must be BGR")
    policy = G1DPPolicy(args.checkpoint, device=args.device,
                        metadata_path=args.task_config, n_samples=1)
    policy.warmup()
    print(json.dumps(policy.contract(), indent=2))
    if args.check_checkpoint:
        return
    if args.mock:
        frames = {k: np.zeros((1, *policy.img_hw[k], 3), dtype=np.uint8) for k in policy.rgb_keys}
        robot = MockRobot(frames, np.asarray(config["init_pose"], dtype=np.float32), fps=policy.fps)
        # Synthetic limits are for exercising the mock only, never real hardware.
        limits = np.tile([-np.pi, np.pi], (14, 1))
    else:
        from .g1_real import UnitreeG1Robot
        with open(args.robot_config) as stream:
            robot_config = json.load(stream)
        limits = urdf_arm_limits(args.urdf)
        robot_config["auto_release"] = args.release_on_exit
        robot_config["fps"] = policy.fps
        robot = UnitreeG1Robot(**robot_config)
    output = Path(args.out or f"outputs/g1/{config['task']}_{time.time_ns()}").expanduser()
    run_session(robot, policy, config, limits, output, args.seconds, mock=args.mock,
                video=not args.no_video, init_seconds=args.init_seconds)


if __name__ == "__main__":
    main()
