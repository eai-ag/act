import unittest
import numpy as np
import rewards


class TestTimePenalty(unittest.TestCase):
    """Unit tests for the time_penalty reward function."""

    def setUp(self):
        """Create simple test trajectories for manual verification."""
        # Simple test trajectories for manual verification
        self.qpos_3 = np.ones((3, 7))  # 3 timesteps, 7 joints
        self.qpos_5 = np.ones((5, 7))  # 5 timesteps, 7 joints
        self.actions_3 = np.zeros((3, 7))
        self.actions_5 = np.zeros((5, 7))
        self.episode_id = 42

    def test_shape_matches_input_length(self):
        """Manual check: output length should equal input trajectory length."""
        result = rewards.time_penalty(self.qpos_3, self.actions_3, self.episode_id)
        # Expected: 3 values (one per timestep)
        self.assertEqual(result.shape, (3,))
        self.assertEqual(len(result), 3)

        # Test with different length
        result_5 = rewards.time_penalty(self.qpos_5, self.actions_5, self.episode_id)
        self.assertEqual(result_5.shape, (5,))
        self.assertEqual(len(result_5), 5)

    def test_all_values_are_one(self):
        """Manual check: each timestep gets reward +1."""
        result = rewards.time_penalty(self.qpos_5, self.actions_5, self.episode_id)
        # Expected: [1, 1, 1, 1, 1] for 5 timesteps
        np.testing.assert_array_equal(result, np.ones(5))

    def test_independent_of_qpos_actions(self):
        """Verify function ignores actual qpos/actions values."""
        qpos_random = np.random.randn(4, 7)
        actions_random = np.random.randn(4, 7)
        result = rewards.time_penalty(qpos_random, actions_random, self.episode_id)
        # Expected: [1, 1, 1, 1] regardless of qpos/actions values
        np.testing.assert_array_equal(result, np.ones(4))

    def test_independent_of_episode_id(self):
        """Verify function ignores episode_id value."""
        result_42 = rewards.time_penalty(self.qpos_3, self.actions_3, 42)
        result_99 = rewards.time_penalty(self.qpos_3, self.actions_3, 99)
        result_0 = rewards.time_penalty(self.qpos_3, self.actions_3, 0)
        # All should be [1, 1, 1]
        np.testing.assert_array_equal(result_42, np.ones(3))
        np.testing.assert_array_equal(result_99, np.ones(3))
        np.testing.assert_array_equal(result_0, np.ones(3))

    def test_single_timestep_trajectory(self):
        """Edge case: trajectory with only one timestep."""
        qpos_1 = np.ones((1, 7))
        actions_1 = np.zeros((1, 7))
        result = rewards.time_penalty(qpos_1, actions_1, self.episode_id)
        # Expected: [1] (single value)
        np.testing.assert_array_equal(result, np.ones(1))


if __name__ == "__main__":
    unittest.main()
