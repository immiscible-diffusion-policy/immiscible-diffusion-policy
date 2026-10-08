import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from examples.training_step import TinyDenoiser, diffusion_loss
from immiscible_diffusion_policy import match_noise


class TrainingStepTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4)
        self.actions = torch.randn(8, 4, 2)
        self.obs = torch.randn(8, 3)
        self.noise = torch.randn_like(self.actions)
        self.t = torch.randint(10, (8,))
        self.alpha_bar = (1 - torch.linspace(1e-4, 0.02, 10)).cumprod(0)
        self.model = TinyDenoiser(4, 2, 3, 10)

    def loss(self, **kwargs):
        return diffusion_loss(self.model, self.actions, self.obs, self.noise,
                              self.t, self.alpha_bar, **kwargs)

    def reference(self, noise, target):
        alpha = self.alpha_bar[self.t].reshape(-1, 1, 1)
        noisy = alpha.sqrt() * self.actions + (1 - alpha).sqrt() * noise
        return F.mse_loss(self.model(noisy, self.t, self.obs), target)

    def test_disabled_and_warmup_are_exactly_vanilla(self):
        expected = self.reference(self.noise, self.noise)
        with patch("examples.training_step.match_noise", side_effect=AssertionError("unexpected assignment")):
            self.assertTrue(torch.equal(self.loss(enabled=False, epoch=100), expected))
            self.assertTrue(torch.equal(self.loss(enabled=True, epoch=49, start_epoch=50), expected))

    def test_activation_boundary_and_epsilon_target(self):
        assigned = match_noise(self.actions, self.noise)
        self.assertFalse(torch.equal(assigned, self.noise))
        expected = self.reference(assigned, assigned)
        self.assertTrue(torch.equal(self.loss(epoch=50, start_epoch=50), expected))
        self.assertTrue(torch.equal(self.loss(epoch=0, start_epoch=0), expected))

    def test_sample_prediction_keeps_clean_target(self):
        assigned = match_noise(self.actions, self.noise)
        expected = self.reference(assigned, self.actions)
        self.assertTrue(torch.equal(self.loss(epoch=50, prediction_type="sample"), expected))

    def test_both_targets_backpropagate_and_update(self):
        for prediction_type in ["epsilon", "sample"]:
            with self.subTest(prediction_type=prediction_type):
                optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
                before = [p.detach().clone() for p in self.model.parameters()]
                optimizer.zero_grad()
                loss = self.loss(epoch=50, prediction_type=prediction_type)
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                for parameter in self.model.parameters():
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all())
                optimizer.step()
                self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, self.model.parameters())))

    def test_invalid_epoch_or_prediction_type(self):
        for kwargs in [{"epoch": -1}, {"start_epoch": -1}, {"prediction_type": "unknown"}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.loss(**kwargs)


if __name__ == "__main__":
    unittest.main()
