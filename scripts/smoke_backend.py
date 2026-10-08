"""Exercise real training, resume, checkpoint loading and inference on temporary data."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    import numpy as np
    import zarr

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backend", choices=["dp-lowdim", "dp-rgb", "dp3"])
    parser.add_argument("--vanilla", action="store_true")
    parser.add_argument("--accumulation", type=int, default=1)
    parser.add_argument("--rollout", action="store_true", help="Also run a short Push-T simulator rollout (DP only)")
    args = parser.parse_args()
    if args.accumulation < 1 or (args.rollout and args.backend == "dp3"):
        parser.error("accumulation must be positive; rollout is supported only for DP")
    root = Path(__file__).resolve().parents[1]
    folder = root / "backends" / ("dp3" if args.backend == "dp3" else "dp")
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMBA_NUM_THREADS="1",
               SDL_VIDEODRIVER="dummy", WANDB_MODE="disabled", PYTHONDONTWRITEBYTECODE="1")
    # Use the copied backend even if a research version is installed editable.
    env["PYTHONPATH"] = str(folder) + os.pathsep + str(root)
    with tempfile.TemporaryDirectory(prefix="idp-backend-smoke-") as scratch:
        scratch = Path(scratch)
        dataset_path = scratch / "synthetic.zarr"
        replay = zarr.open_group(str(dataset_path), mode="w")
        data = replay.create_group("data")
        meta = replay.create_group("meta")
        n = 8 * 12
        rng = np.random.default_rng(12)
        meta.create_dataset("episode_ends", data=np.arange(12, n + 1, 12, dtype=np.int64))
        action_dim = 14 if args.backend == "dp3" else 2
        data.create_dataset("action", data=rng.normal(size=(n, action_dim)).astype(np.float32))
        if args.backend == "dp3":
            data.create_dataset("state", data=rng.normal(size=(n, 14)).astype(np.float32))
            data.create_dataset("point_cloud", data=rng.normal(size=(n, 32, 6)).astype(np.float32))
        else:
            data.create_dataset("state", data=rng.uniform(100, 400, size=(n, 5)).astype(np.float32))
            data.create_dataset("keypoint", data=rng.uniform(100, 400, size=(n, 9, 2)).astype(np.float32))
            data.create_dataset("img", data=rng.integers(0, 256, (n, 96, 96, 3), dtype=np.uint8))

        config = {"dp-lowdim": "immiscible_pusht_lowdim",
                  "dp-rgb": "immiscible_pusht_rgb", "dp3": "immiscible_dp3"}[args.backend]
        command = [sys.executable, "train_dp3.py" if args.backend == "dp3" else "train.py",
                   "--config-name=" + config,
                   "training.device=cpu", "training.num_epochs=1", "training.resume=true",
                   "training.max_train_steps=2", "training.max_val_steps=1",
                   "training.gradient_accumulate_every=" + str(args.accumulation),
                   "training.checkpoint_every=1", "training.val_every=1",
                   "training.lr_warmup_steps=0", "training.tqdm_interval_sec=100",
                   "dataloader.batch_size=2", "dataloader.num_workers=0",
                   "dataloader.pin_memory=false", "dataloader.persistent_workers=false",
                   "val_dataloader.batch_size=2", "val_dataloader.num_workers=0",
                   "val_dataloader.pin_memory=false", "val_dataloader.persistent_workers=false",
                   "horizon=8", "n_obs_steps=2", "n_action_steps=4",
                   "policy.num_inference_steps=2", "policy.noise_scheduler.num_train_timesteps=10",
                   "policy.immiscible_start_epoch=1",
                   "policy.use_immiscible_diffusion=" + str(not args.vanilla).lower(),
                   "task.dataset.val_ratio=0.25", "task.dataset.max_train_episodes=null",
                   "logging.mode=disabled", "hydra.run.dir=" + str(scratch / "run")]
        if args.backend == "dp3":
            command += ["dataset_path=" + str(dataset_path), "policy.down_dims=[16,32]",
                        "policy.diffusion_step_embed_dim=16", "policy.encoder_output_dim=16",
                        "task.shape_meta.obs.point_cloud.shape=[32,6]", "checkpoint.save_ckpt=true"]
        else:
            command += ["task.dataset.zarr_path=" + str(dataset_path), "training.enable_rollouts=false",
                        "checkpoint.topk.k=0", "checkpoint.save_last_snapshot=false"]
            prefix = "policy.model" if args.backend == "dp-lowdim" else "policy"
            command += [prefix + ".down_dims=[16,32]", prefix + ".diffusion_step_embed_dim=16"]
            if args.rollout:
                command += ["task.env_runner.n_train=0", "task.env_runner.n_test=1",
                            "task.env_runner.n_train_vis=0", "task.env_runner.n_test_vis=0",
                            "task.env_runner.n_envs=1", "task.env_runner.max_steps=4"]

        def run(cmd):
            result = subprocess.run(cmd, cwd=folder, env=env, text=True, capture_output=True)
            if result.returncode:
                print(result.stdout)
                print(result.stderr, file=sys.stderr)
                raise RuntimeError(f"command failed: {cmd}")
            return result.stdout

        run(command)
        checkpoint = scratch / "run/checkpoints/latest.ckpt"
        assert checkpoint.is_file(), "training did not write latest.ckpt"
        resumed = ["training.num_epochs=2" if s == "training.num_epochs=1" else s for s in command]
        run(resumed)
        records = [json.loads(line) for line in (scratch / "run/logs.json.txt").read_text().splitlines()]
        epochs = [r for r in records if "assignment_active" in r]
        assert [r["epoch"] for r in epochs] == [0, 1], epochs
        assert [r["assignment_active"] for r in epochs] == [False, not args.vanilla], epochs
        import dill
        import torch
        with checkpoint.open("rb") as stream:
            payload = torch.load(stream, map_location="cpu", pickle_module=dill)
        counts = {k: dill.loads(payload["pickles"][k])
                  for k in ["optimizer_step", "ema_optimization_step", "global_step"]}
        expected_updates = 2 * ((2 + args.accumulation - 1) // args.accumulation)
        assert counts["optimizer_step"] == counts["ema_optimization_step"] == expected_updates, counts
        assert counts["global_step"] == 3, counts  # zero-based final batch index
        metric_path = scratch / "metrics.json"
        run([sys.executable, "eval.py" if args.backend == "dp3" else "eval_offline.py",
             "--checkpoint", str(checkpoint), "--dataset", str(dataset_path),
             "--batch-size", "2", "--max-batches", "1", "--output", str(metric_path)])
        metrics = json.loads(metric_path.read_text())
        assert metrics["epoch"] == 1 and metrics["windows"] == 2, metrics
        assert np.isfinite(metrics["mean_squared_error"]), metrics
        if args.rollout:
            run([sys.executable, "eval.py", "--checkpoint", str(checkpoint),
                 "--output_dir", str(scratch / "rollout"), "--device", "cpu"])
            rollout = json.loads((scratch / "rollout/eval_log.json").read_text())
            assert "test/mean_score" in rollout and np.isfinite(rollout["test/mean_score"]), rollout
        print(json.dumps({"backend": args.backend, "vanilla": args.vanilla,
                          "status": "passed", "epochs": [0, 1],
                          "assignment_active": [r["assignment_active"] for r in epochs],
                          "checkpoint_counters": counts,
                          "short_pusht_rollout": args.rollout,
                          "offline_evaluation": metrics}, indent=2))


if __name__ == "__main__":
    main()
