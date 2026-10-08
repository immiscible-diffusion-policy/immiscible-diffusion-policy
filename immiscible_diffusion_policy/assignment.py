"""Hungarian action-noise assignment, extracted from the experiment policies.

Actions are already dataset-normalized. Batch standardization is used only
to choose the assignment; it does not change the actions being corrupted.
See THIRD_PARTY_NOTICES.md for source provenance and attribution.
"""

import torch
from scipy.optimize import linear_sum_assignment


@torch.no_grad()
def assignment_indices(actions: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    """Return one noise-row index per action row, without drawing new noise.

    Args:
        actions: Floating tensor [B, T, Da], or [B, Da], with B >= 1.
        noise: Floating tensor with matching batch/time axes and D >= Da
            channels. Only its first Da channels enter the cost. Additional
            channels support policies that jointly denoise actions and features.

    Returns:
        A permutation of 0..B-1, as torch.long on the input device.

    Matching uses float32, population standard deviation clamped to 1e-6,
    and unsquared Euclidean distance, as in the experiment implementation.
    SciPy solves the assignment on CPU; GPU inputs require synchronization.
    The explicit pairwise difference uses O(B**2 * T * Da) working memory.
    """
    if actions.ndim not in (2, 3) or noise.ndim != actions.ndim:
        raise ValueError("actions and noise must both have rank 2 or rank 3")
    if actions.shape[:-1] != noise.shape[:-1]:
        raise ValueError("actions and noise must have matching batch/time axes")
    if any(size == 0 for size in actions.shape):
        raise ValueError("action dimensions must be nonempty")
    if noise.shape[-1] < actions.shape[-1]:
        raise ValueError("noise must contain at least the action channels")
    if actions.device != noise.device:
        raise ValueError("actions and noise must be on the same device")
    if not actions.is_floating_point() or not noise.is_floating_point():
        raise TypeError("actions and noise must be floating-point tensors")

    traj_points = actions.flatten(start_dim=1).to(torch.float32)
    noise_points = noise[..., :actions.shape[-1]].flatten(start_dim=1).to(torch.float32)
    if not torch.isfinite(traj_points).all() or not torch.isfinite(noise).all():
        raise ValueError("actions and noise must contain only finite values")
    traj_mean = traj_points.mean(dim=0, keepdim=True)
    # unbiased=False is equivalent to correction=0 and supports PyTorch 1.12.
    traj_std = traj_points.std(dim=0, keepdim=True, unbiased=False).clamp(min=1e-6)
    traj_points_standardized = (traj_points - traj_mean) / traj_std
    distance_points = traj_points_standardized.unsqueeze(1) - noise_points.unsqueeze(0)
    distance = torch.linalg.vector_norm(distance_points, dim=2)
    if not torch.isfinite(distance).all():
        raise ValueError("matching cost is non-finite after float32 conversion")
    _, col_ind = linear_sum_assignment(distance.cpu().numpy())
    return torch.as_tensor(col_ind, dtype=torch.long, device=noise.device)


def match_noise(actions: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    """Permute complete noise rows using the action-based assignment.

    Inputs are not modified. Output shape, dtype, device, and the noise pool
    are preserved. This is a training operation; inference is unchanged.
    """
    return noise[assignment_indices(actions, noise)]
