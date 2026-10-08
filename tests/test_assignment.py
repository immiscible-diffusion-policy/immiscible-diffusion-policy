import itertools
import unittest
from unittest.mock import patch

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from immiscible_diffusion_policy import assignment_indices, match_noise


def original_experiment_assignment(actions, raw_noise):
    """Reference copied from the experiment policies; see THIRD_PARTY_NOTICES.md."""
    traj_points = actions.flatten(start_dim=1).to(torch.float32)
    noise_points = raw_noise[..., :actions.shape[-1]].flatten(start_dim=1).to(torch.float32)
    traj_mean = traj_points.mean(dim=0, keepdim=True)
    traj_std = traj_points.std(dim=0, keepdim=True, unbiased=False).clamp(min=1e-6)
    standardized = (traj_points - traj_mean) / traj_std
    distance = torch.linalg.vector_norm(standardized.unsqueeze(1) - noise_points.unsqueeze(0), dim=2)
    _, col_ind = linear_sum_assignment(distance.cpu().detach().numpy())
    return raw_noise[col_ind]


class AssignmentTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.actions = torch.randn(8, 4, 2)
        self.noise = torch.randn_like(self.actions)

    def test_agrees_with_experiment_code(self):
        for seed in range(5):
            torch.manual_seed(seed)
            for shape in [(16, 16, 2), (8, 8, 14), (8, 32, 16)]:
                with self.subTest(seed=seed, shape=shape):
                    actions = torch.randn(shape)
                    noise = torch.randn_like(actions)
                    self.assertTrue(torch.equal(match_noise(actions, noise),
                                                original_experiment_assignment(actions, noise)))

    def test_permutation_preserves_pool_and_inputs(self):
        actions_before, noise_before = self.actions.clone(), self.noise.clone()
        indices = assignment_indices(self.actions, self.noise)
        matched = match_noise(self.actions, self.noise)
        self.assertTrue(torch.equal(indices.sort().values, torch.arange(8)))
        self.assertTrue(torch.equal(matched[indices.argsort()], self.noise))
        self.assertTrue(torch.equal(self.actions, actions_before))
        self.assertTrue(torch.equal(self.noise, noise_before))

    def test_unsquared_euclidean_global_minimum(self):
        # Find a small case where squaring the costs changes the optimum.
        rng = np.random.default_rng(9)
        for _ in range(1000):
            actions = rng.normal(size=(4, 3)).astype(np.float32)
            noise = rng.normal(size=(4, 3)).astype(np.float32)
            z = (actions - actions.mean(axis=0)) / actions.std(axis=0)
            cost = np.linalg.norm(z[:, None] - noise[None], axis=-1)
            permutations = list(itertools.permutations(range(4)))
            best = min(permutations, key=lambda p: cost[np.arange(4), p].sum())
            squared_best = min(permutations, key=lambda p: (cost[np.arange(4), p] ** 2).sum())
            if best != squared_best:
                break
        else:
            self.fail("failed to find the fixed-seed unsquared-cost regression case")
        actual = assignment_indices(torch.from_numpy(actions), torch.from_numpy(noise))
        self.assertEqual(tuple(actual.tolist()), best)

    def test_constant_coordinates_and_singleton(self):
        actions = self.actions.clone()
        actions[..., 0] = 3
        self.assertTrue(torch.equal(match_noise(actions, self.noise),
                                    original_experiment_assignment(actions, self.noise)))
        self.assertTrue(torch.equal(match_noise(torch.ones(1, 4, 2), self.noise[:1]), self.noise[:1]))
        self.assertEqual(assignment_indices(torch.ones_like(actions), self.noise).unique().numel(), 8)

    def test_extra_channels_move_with_complete_row(self):
        extra = torch.arange(8).reshape(8, 1, 1).expand(8, 4, 1).float()
        noise = torch.cat([self.noise, extra], dim=-1)
        indices = assignment_indices(self.actions, self.noise)
        self.assertTrue(torch.equal(match_noise(self.actions, noise), noise[indices]))

    def test_matching_consumes_no_randomness(self):
        state = torch.random.get_rng_state()
        match_noise(self.actions, self.noise)
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))

    def test_output_retains_dtype(self):
        noise = self.noise.double()
        self.assertEqual(match_noise(self.actions, noise).dtype, torch.float64)

    def test_invalid_inputs(self):
        cases = [
            (self.actions[:0], self.noise[:0]),
            (self.actions, self.noise[:, :2]),
            (self.actions, self.noise[..., :1]),
            (self.actions.flatten(), self.noise.flatten()),
            (self.actions * float("nan"), self.noise),
            (self.actions, self.noise * float("inf")),
        ]
        for actions, noise in cases:
            with self.subTest(shape=actions.shape), self.assertRaises(ValueError):
                match_noise(actions, noise)
        with self.assertRaises(TypeError):
            match_noise(self.actions.long(), self.noise)

    def test_discrete_solver_runs_without_gradients(self):
        actions = self.actions.clone().requires_grad_()
        with patch("immiscible_diffusion_policy.assignment.linear_sum_assignment",
                   wraps=linear_sum_assignment) as solve:
            match_noise(actions, self.noise)
        self.assertIsInstance(solve.call_args.args[0], np.ndarray)
        self.assertIsNone(actions.grad)


if __name__ == "__main__":
    unittest.main()
