"""Record the internal controller's stationary pose from rt/lowstate; publishes no commands."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
from .g1_real import _import_robot_stack

def _read_lowstate_arm_q(iface: str, seconds: float = 1.0, timeout: float = 20.0):
    """Subscribe to rt/lowstate only (no publisher), average `seconds` of samples."""
    arm_mod, _, ChannelFactoryInitialize = _import_robot_stack()
    from unitree_sdk2py.core.channel import ChannelSubscriber
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

    ChannelFactoryInitialize(0, iface)
    sub = ChannelSubscriber("rt/lowstate", LowState_)
    sub.Init()
    arm_idx = [int(j) for j in arm_mod.G1_29_JointArmIndex]
    waist_idx = int(arm_mod.G1_29_JointIndex.kWaistYaw)
    n_motors = len(list(arm_mod.G1_29_JointIndex))
    samples, all_q = [], []
    t0 = time.time()
    t_first = None
    while True:
        msg = sub.Read()
        now = time.time()
        if msg is not None:
            q = np.array([msg.motor_state[i].q for i in range(n_motors)], dtype=np.float32)
            samples.append(q[arm_idx]); all_q.append(q)
            t_first = t_first or now
            if now - t_first >= seconds:
                break
        elif now - t0 > timeout:
            sys.exit("no rt/lowstate: robot off, remote sequence not done (L2+B -> L2+UP -> R1+Y), or wrong --iface")
        time.sleep(0.005)
    S = np.stack(samples); A = np.stack(all_q)
    return dict(arm_q=S.mean(0), arm_q_std=S.std(0), all_motor_q=A.mean(0), waist_yaw=float(A[:, waist_idx].mean()),
                n_samples=len(S))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iface", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    output = Path(args.out).expanduser()
    if output.exists():
        parser.error("Output already exists; choose a new file")
    result = _read_lowstate_arm_q(args.iface)
    if result["arm_q_std"].max() > 0.02:
        raise RuntimeError("Arms moved while sampling; capture again while stationary")
    doc = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in result.items()}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(doc, stream, indent=2)
    print(output)

if __name__ == "__main__":
    main()
