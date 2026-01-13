import unittest
import torch
import numpy as np
import h5py
from utils import get_norm_stats
import os


def original_get_norm_stats(dataset_dir, num_episodes):
    """
    Original version of get_norm_stats before adapting for variable lengths.
    Uses torch.stack (assumes uniform lengths) and mean/std across dim=[0, 1].
    https://github.com/tonyzhaozh/act/blob/742c753c0d4a5d87076c8f69e5628c79a8cc5488/utils.py#L79
    """
    all_qpos_data = []
    all_action_data = []
    for episode_idx in range(num_episodes):
        dataset_path = os.path.join(dataset_dir, f"episode_{episode_idx}.hdf5")
        with h5py.File(dataset_path, "r") as root:
            qpos = root["/observations/qpos"][()]
            qvel = root["/observations/qvel"][()]
            action = root["/action"][()]
        all_qpos_data.append(torch.from_numpy(qpos))
        all_action_data.append(torch.from_numpy(action))
    all_qpos_data = torch.stack(all_qpos_data)
    all_action_data = torch.stack(all_action_data)
    all_action_data = all_action_data

    # normalize action data
    action_mean = all_action_data.mean(dim=[0, 1], keepdim=True)
    action_std = all_action_data.std(dim=[0, 1], keepdim=True)
    action_std = torch.clip(action_std, 1e-2, np.inf)  # clipping

    # normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=[0, 1], keepdim=True)
    qpos_std = all_qpos_data.std(dim=[0, 1], keepdim=True)
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf)  # clipping

    stats = {
        "action_mean": action_mean.numpy().squeeze(),
        "action_std": action_std.numpy().squeeze(),
        "qpos_mean": qpos_mean.numpy().squeeze(),
        "qpos_std": qpos_std.numpy().squeeze(),
        "example_qpos": qpos,
    }

    return stats


class TestNormStats(unittest.TestCase):
    def test_get_norm_stats_sim_vs_original(self):
        """
        Compare current get_norm_stats (with cat) vs original (with stack) on uniform-length sim data.
        Ensures no regression and that the change is equivalent on fixed lengths.
        """
        dataset_dir = "data/sim_transfer_cube_human"
        num_episodes = 16

        # Current implementation
        current_stats = get_norm_stats(dataset_dir, num_episodes)

        # Original implementation (mock)
        original_stats = original_get_norm_stats(dataset_dir, num_episodes)

        # Assert means and stds are close (should be identical on uniform data)
        np.testing.assert_allclose(
            current_stats["action_mean"], original_stats["action_mean"], atol=1e-6
        )
        np.testing.assert_allclose(
            current_stats["action_std"], original_stats["action_std"], atol=1e-6
        )
        np.testing.assert_allclose(
            current_stats["qpos_mean"], original_stats["qpos_mean"], atol=1e-6
        )
        np.testing.assert_allclose(
            current_stats["qpos_std"], original_stats["qpos_std"], atol=1e-6
        )


if __name__ == "__main__":
    unittest.main()
