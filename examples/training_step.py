"""Run one CPU training step on generated tensors, without data or checkpoints."""

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from immiscible_diffusion_policy import match_noise


class TinyDenoiser(nn.Module):
    """Small conditional network for demonstrating the training interface."""

    def __init__(self, horizon, action_dim, obs_dim, num_diffusion_steps):
        super().__init__()
        self.num_diffusion_steps = num_diffusion_steps
        size = horizon * action_dim
        self.net = nn.Sequential(nn.Linear(size + obs_dim + 1, 64), nn.SiLU(), nn.Linear(64, size))

    def forward(self, noisy_actions, timesteps, observations):
        time = timesteps.to(noisy_actions.dtype).unsqueeze(1) / self.num_diffusion_steps
        features = torch.cat([noisy_actions.flatten(1), observations, time], dim=1)
        return self.net(features).reshape_as(noisy_actions)


def diffusion_loss(model, actions, observations, noise, timesteps, alpha_bar,
                   *, enabled=True, epoch=0, start_epoch=50, prediction_type="epsilon"):
    """Apply assignment after warm-up, then the usual diffusion regression loss.

    In a real policy, normalize actions using the dataset normalizer before
    calling this function. Supply the policy's existing schedule and model.
    Noise and timesteps are supplied explicitly to make comparisons repeatable.
    """
    if start_epoch < 0 or epoch < 0:
        raise ValueError("epoch and start_epoch must be nonnegative")
    if enabled and epoch >= start_epoch:
        noise = match_noise(actions, noise)
    alpha = alpha_bar[timesteps].reshape(-1, 1, 1)
    noisy_actions = alpha.sqrt() * actions + (1 - alpha).sqrt() * noise
    prediction = model(noisy_actions, timesteps, observations)
    if prediction_type == "epsilon":
        target = noise
    elif prediction_type == "sample":
        target = actions
    else:
        raise ValueError("prediction_type must be 'epsilon' or 'sample'")
    return F.mse_loss(prediction, target)


def run_step(config):
    torch.manual_seed(config["seed"])
    torch.set_num_threads(1)
    shape = (config["batch_size"], config["horizon"], config["action_dim"])
    # Illustrative two-cluster actions; no demonstration data is loaded.
    modes = 2 * (torch.arange(shape[0]) % 2).float() - 1
    actions = 0.5 * modes[:, None, None] + 0.1 * torch.randn(shape)
    observations = torch.randn(shape[0], config["obs_dim"])
    model = TinyDenoiser(shape[1], shape[2], config["obs_dim"], config["num_diffusion_steps"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    noise = torch.randn_like(actions)
    timesteps = torch.randint(config["num_diffusion_steps"], (shape[0],))
    # An illustrative DDPM schedule; integrations retain their own schedules.
    alpha_bar = (1 - torch.linspace(1e-4, 0.02, config["num_diffusion_steps"])).cumprod(0)
    loss = diffusion_loss(
        model, actions, observations, noise, timesteps, alpha_bar,
        enabled=config["use_immiscible_diffusion"], epoch=config["epoch"],
        start_epoch=config["immiscible_start_epoch"], prediction_type=config["prediction_type"],
    )
    optimizer.zero_grad()
    loss.backward()
    grad_norm = sum(p.grad.square().sum() for p in model.parameters()).sqrt()
    optimizer.step()
    return {
        "loss": loss.item(), "gradient_norm": grad_norm.item(),
        "assignment_active": bool(config["use_immiscible_diffusion"]
                                  and config["epoch"] >= config["immiscible_start_epoch"]),
        "prediction_type": config["prediction_type"], "device": "cpu",
    }


def main():
    import yaml

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=Path(__file__).resolve().parents[1] / "configs/example.yaml")
    parser.add_argument("--epoch", type=int)
    parser.add_argument("--prediction-type", choices=["epsilon", "sample"])
    parser.add_argument("--vanilla", action="store_true")
    args = parser.parse_args()
    with args.config.open() as stream:
        config = yaml.safe_load(stream)
    if args.epoch is not None:
        config["epoch"] = args.epoch
    if args.prediction_type is not None:
        config["prediction_type"] = args.prediction_type
    if args.vanilla:
        config["use_immiscible_diffusion"] = False
    print(json.dumps(run_step(config), indent=2))


if __name__ == "__main__":
    main()
