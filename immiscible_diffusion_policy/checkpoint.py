"""Load a trained bundled policy for inference (requires backend dependencies)."""

def load_policy(path, device="cpu", use_ema=True):
    """Return (policy, config, completed_epoch) from a trusted checkpoint.

    Install the corresponding bundled backend first. The saved normalizer is
    restored together with the model; callers pass observations in raw dataset
    units, with the same keys, shapes and image scaling used during training.
    """
    import dill
    import hydra
    import torch
    from omegaconf import OmegaConf

    OmegaConf.register_new_resolver("eval", eval, replace=True)
    with open(path, "rb") as stream:
        payload = torch.load(stream, map_location="cpu", pickle_module=dill)
    cfg = payload["cfg"]
    # Resolve only values used by the policy/dataset. Historical checkpoints
    # can retain unrelated Hydra runtime interpolations in output paths.
    policy = hydra.utils.instantiate(cfg.policy)
    states = payload["state_dicts"]
    key = "ema_model" if use_ema and cfg.training.use_ema else "model"
    policy.load_state_dict(states[key])
    policy.to(device).eval()
    epoch = int(dill.loads(payload["pickles"]["epoch"]))
    return policy, cfg, epoch
