"""Offline action prediction metrics for bundled DP and DP3 checkpoints."""

def main():
    import argparse
    import json
    from pathlib import Path

    import hydra
    import torch
    from torch.utils.data import DataLoader
    from immiscible_diffusion_policy.checkpoint import load_policy

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", help="Override the saved Zarr path")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--split", choices=["validation", "train"], default="validation")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--raw", action="store_true", help="Use raw weights instead of EMA")
    parser.add_argument("--output", type=Path, help="Optional JSON output; refuses to overwrite")
    args = parser.parse_args()
    if args.batch_size < 1 or (args.max_batches is not None and args.max_batches < 1):
        parser.error("batch-size and max-batches must be positive")
    torch.manual_seed(args.seed)
    policy, cfg, epoch = load_policy(args.checkpoint, args.device, use_ema=not args.raw)
    if args.dataset:
        cfg.task.dataset.zarr_path = str(Path(args.dataset).expanduser().resolve())
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    if args.split == "validation":
        dataset = dataset.get_validation_dataset()
    if len(dataset) == 0:
        parser.error("selected dataset split is empty")

    def to_device(value):
        if isinstance(value, dict):
            return {k: to_device(v) for k, v in value.items()}
        return value.to(args.device)

    squared, absolute, elements, windows = 0.0, 0.0, 0, 0
    with torch.no_grad():
        for index, batch in enumerate(DataLoader(dataset, batch_size=args.batch_size, shuffle=False)):
            batch = to_device(batch)
            obs = batch["obs"] if isinstance(batch["obs"], dict) else {"obs": batch["obs"]}
            prediction = policy.predict_action(obs)["action_pred"]
            target = batch["action"]
            if getattr(policy, "pred_action_steps_only", False):
                start = policy.n_obs_steps - int(policy.oa_step_convention)
                target = target[:, start:start + policy.n_action_steps]
            if prediction.shape != target.shape:
                raise ValueError(f"prediction {prediction.shape} does not match target {target.shape}")
            difference = (prediction - target).double()
            if not torch.isfinite(difference).all():
                raise ValueError("non-finite action predictions or targets")
            squared += difference.square().sum().item()
            absolute += difference.abs().sum().item()
            elements += difference.numel()
            windows += difference.shape[0]
            if args.max_batches is not None and index + 1 >= args.max_batches:
                break
    metrics = dict(epoch=epoch, split=args.split, seed=args.seed, windows=windows,
                   mean_squared_error=squared / elements, mean_absolute_error=absolute / elements,
                   weights="raw" if args.raw or not cfg.training.use_ema else "ema")
    text = json.dumps(metrics, indent=2, allow_nan=False)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            stream.write(text + "\n")


if __name__ == "__main__":
    main()
