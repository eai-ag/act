import unittest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
import h5py

from compute_value_targets import discount_cumsum, ComputeValueTargets


class TestDiscountCumsum(unittest.TestCase):
    """Unit tests for discount_cumsum helper function (manual verification possible)."""

    def test_simple_case_gamma_0_5(self):
        """
        Manual verification:
        x = [1, 2, 3], gamma = 0.5
        Expected: [1 + 0.5*2 + 0.25*3, 2 + 0.5*3, 3]
                 = [1 + 1 + 0.75, 2 + 1.5, 3]
                 = [2.75, 3.5, 3.0]
        """
        x = np.array([1.0, 2.0, 3.0])
        result = discount_cumsum(x, discount=0.5)
        expected = np.array([2.75, 3.5, 3.0])
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_gamma_1_0_cumulative_sum(self):
        """
        Manual verification:
        x = [1, 2, 3], gamma = 1.0
        Expected: [1+2+3, 2+3, 3] = [6, 5, 3]
        """
        x = np.array([1.0, 2.0, 3.0])
        result = discount_cumsum(x, discount=1.0)
        expected = np.array([6.0, 5.0, 3.0])
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_gamma_0_0_original_values(self):
        """
        Manual verification:
        x = [1, 2, 3], gamma = 0.0
        Expected: [1, 2, 3] (no future discounting)
        """
        x = np.array([1.0, 2.0, 3.0])
        result = discount_cumsum(x, discount=0.0)
        expected = np.array([1.0, 2.0, 3.0])
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_single_element_array(self):
        """
        Manual verification:
        x = [5], gamma = 0.99
        Expected: [5] (no future to discount)
        """
        x = np.array([5.0])
        result = discount_cumsum(x, discount=0.99)
        expected = np.array([5.0])
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_negative_values_gamma_0_5(self):
        """
        Manual verification:
        x = [-1, -2, -3], gamma = 0.5
        Expected: [-1 + 0.5*(-2) + 0.25*(-3), -2 + 0.5*(-3), -3]
                 = [-1 -1 -0.75, -2 -1.5, -3]
                 = [-2.75, -3.5, -3.0]
        """
        x = np.array([-1.0, -2.0, -3.0])
        result = discount_cumsum(x, discount=0.5)
        expected = np.array([-2.75, -3.5, -3.0])
        np.testing.assert_allclose(result, expected, rtol=1e-10)


class TestComputeRewards(unittest.TestCase):
    """Tests for compute_rewards method with mocked dependencies."""

    def setUp(self):
        # Minimal test config matching rl_cfg.yaml structure
        self.test_config = {
            "data_dir": "/fake/path",
            "rl_params": {"gamma": 0.5},  # Use 0.5 for easy manual calculation
            "reward_pipeline": [{"name": "time_penalty", "weight": -1.0}],
            "step1_write_to_disk": ["rew_total", "v_target"],
        }

        # Simple synthetic trajectory (3 timesteps, 7 dims) for manual verification
        self.qpos = np.ones((3, 7))
        self.actions = np.zeros((3, 7))
        self.episode_id = 1

        # Manual calculation for verification:
        # time_penalty → [1,1,1]
        # weight = -1.0 → rew_total = [-1,-1,-1]
        self.expected_rew_total = np.array([-1.0, -1.0, -1.0])

    @patch("compute_value_targets.rewards.time_penalty")
    def test_compute_rewards_single_component(self, mock_time_penalty):
        """Test compute_rewards with mocked reward function."""
        # Setup mock to return +1 per timestep
        mock_time_penalty.return_value = np.ones(3)

        # Create instance with config dict
        computer = ComputeValueTargets.__new__(ComputeValueTargets)
        computer.cfg = self.test_config
        computer.gamma = 0.5

        # Compute rewards
        result = computer.compute_rewards(self.qpos, self.actions, self.episode_id)

        # Verify mock was called correctly
        mock_time_penalty.assert_called_once_with(
            qpos=self.qpos, actions=self.actions, episode_id=self.episode_id
        )

        # Manual verification:
        # time_penalty returns [1,1,1], weight = -1.0
        # rew_total = [-1,-1,-1]
        self.assertIn("time_penalty", result)
        self.assertIn("rew_total", result)
        np.testing.assert_array_equal(result["time_penalty"], np.ones(3))
        np.testing.assert_array_equal(result["rew_total"], self.expected_rew_total)

    @patch("compute_value_targets.rewards.time_penalty")
    def test_compute_rewards_multiple_components(self, mock_time_penalty):
        """Test compute_rewards with multiple reward components (mocked as same function)."""
        # Config with two components (both time_penalty for simplicity)
        config_multi = {
            "data_dir": "/fake/path",
            "rl_params": {"gamma": 0.5},
            "reward_pipeline": [
                {"name": "time_penalty", "weight": 0.5},
                {"name": "time_penalty", "weight": -1.0},
            ],
            "step1_write_to_disk": ["rew_total", "v_target"],
        }

        # Mock returns different values for each call
        mock_time_penalty.side_effect = [
            np.array([1.0, 2.0, 3.0]),  # First call
            np.array([0.5, 0.5, 0.5]),  # Second call
        ]

        computer = ComputeValueTargets.__new__(ComputeValueTargets)
        computer.cfg = config_multi
        computer.gamma = 0.5

        result = computer.compute_rewards(self.qpos, self.actions, self.episode_id)

        # Manual verification:
        # Component 1: [1,2,3] * 0.5 = [0.5, 1.0, 1.5]
        # Component 2: [0.5,0.5,0.5] * -1.0 = [-0.5, -0.5, -0.5]
        # rew_total = [0, 0.5, 1.0]
        expected_total = np.array([0.0, 0.5, 1.0])
        np.testing.assert_allclose(result["rew_total"], expected_total, rtol=1e-10)

    def test_compute_rewards_missing_weight_raises_keyerror(self):
        """Test that reward component without weight field raises KeyError."""
        config_no_weight = {
            "data_dir": "/fake/path",
            "rl_params": {"gamma": 0.5},
            "reward_pipeline": [{"name": "time_penalty"}],  # Missing weight
            "step1_write_to_disk": ["rew_total", "v_target"],
        }

        computer = ComputeValueTargets.__new__(ComputeValueTargets)
        computer.cfg = config_no_weight
        computer.gamma = 0.5

        with self.assertRaises(KeyError) as cm:
            computer.compute_rewards(self.qpos, self.actions, self.episode_id)

        # Verify error message contains 'weight'
        self.assertIn("weight", str(cm.exception))


class TestProcessTrajectory(unittest.TestCase):
    """Integration tests for process_trajectory method."""

    def setUp(self):
        self.test_config = {
            "data_dir": "/fake/path",
            "rl_params": {"gamma": 0.5},
            "reward_pipeline": [{"name": "time_penalty", "weight": -1.0}],
            "step1_write_to_disk": ["rew_total", "v_target"],
        }

        self.qpos = np.ones((3, 7))
        self.actions = np.zeros((3, 7))
        self.episode_id = 42

    @patch("compute_value_targets.rewards.time_penalty")
    def test_process_trajectory_returns_all_fields(self, mock_time_penalty):
        """Verify process_trajectory returns dict with all expected fields."""
        mock_time_penalty.return_value = np.ones(3)

        computer = ComputeValueTargets.__new__(ComputeValueTargets)
        computer.cfg = self.test_config
        computer.gamma = 0.5

        result = computer.process_trajectory(self.qpos, self.actions, self.episode_id)

        # Should contain all components from compute_rewards plus v_target
        expected_keys = {"time_penalty", "rew_total", "v_target"}
        self.assertEqual(set(result.keys()), expected_keys)

    @patch("compute_value_targets.rewards.time_penalty")
    def test_process_trajectory_shapes_match(self, mock_time_penalty):
        """Verify output arrays have same length as input trajectory."""
        mock_time_penalty.return_value = np.ones(3)

        computer = ComputeValueTargets.__new__(ComputeValueTargets)
        computer.cfg = self.test_config
        computer.gamma = 0.5

        result = computer.process_trajectory(self.qpos, self.actions, self.episode_id)

        # All arrays should have length 3 (same as trajectory)
        self.assertEqual(len(result["time_penalty"]), 3)
        self.assertEqual(len(result["rew_total"]), 3)
        self.assertEqual(len(result["v_target"]), 3)

    @patch("compute_value_targets.rewards.time_penalty")
    def test_process_trajectory_values_correct(self, mock_time_penalty):
        """Manual verification of computed values."""
        mock_time_penalty.return_value = np.ones(3)

        computer = ComputeValueTargets.__new__(ComputeValueTargets)
        computer.cfg = self.test_config
        computer.gamma = 0.5

        result = computer.process_trajectory(self.qpos, self.actions, self.episode_id)

        # Manual verification:
        # time_penalty = [1,1,1]
        # rew_total = [-1,-1,-1] (weight -1.0)
        # v_target = discount_cumsum([-1,-1,-1], 0.5) = [-1.75, -1.5, -1.0]
        np.testing.assert_array_equal(result["time_penalty"], np.ones(3))
        np.testing.assert_array_equal(result["rew_total"], np.array([-1.0, -1.0, -1.0]))
        np.testing.assert_allclose(
            result["v_target"], np.array([-1.75, -1.5, -1.0]), rtol=1e-10
        )


class TestRun(unittest.TestCase):
    """Integration tests for the run method with real HDF5 file operations."""

    # Episode ID → trajectory length mapping
    LENGTH_MAP = {0: 360, 1: 376, 2: 428}

    def setUp(self):
        """Create temporary directory and copy real HDF5 files."""
        import shutil
        from pathlib import Path

        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp_dir, ignore_errors=True)

        # Test config
        self.test_config = {
            "data_dir": self.temp_dir,
            "rl_params": {"gamma": 0.5},
            "reward_pipeline": [{"name": "time_penalty", "weight": -1.0}],
            "step1_write_to_disk": ["rew_total", "v_target"],
        }

        # Copy real HDF5 files from source directory
        test_dir = Path(__file__).parent
        source_dir = test_dir.parent / "data" / "tomato_picking"

        # Check source directory exists
        if not source_dir.exists():
            raise FileNotFoundError(
                f"Source directory not found: {source_dir}. "
                f"Required for test data. Please ensure the data files are available."
            )

        self.test_files = []

        for episode_id in [0, 1, 2]:
            source_file = source_dir / f"episode_{episode_id}.hdf5"
            dest_file = os.path.join(self.temp_dir, f"episode_{episode_id}.hdf5")

            if source_file.exists():
                shutil.copy2(str(source_file), dest_file)
                self.test_files.append(dest_file)
            else:
                raise FileNotFoundError(
                    f"Source HDF5 file not found: {source_file}. "
                    f"Required for test data. Please ensure the file exists in {source_dir}."
                )

    def tearDown(self):
        """Clean up temporary files and directory."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_run_processes_all_files(self):
        """Test that run processes all HDF5 files in directory."""
        # Use class-level trajectory lengths for episode_0, episode_1, episode_2

        # Create computer instance with temporary config
        computer = ComputeValueTargets.__new__(ComputeValueTargets)
        computer.cfg = self.test_config
        computer.gamma = 0.99

        # Run processing
        computer.run()

        # Verify all files have the new datasets with correct lengths
        for file_path in self.test_files:
            # Extract episode ID from filename (e.g., "episode_0.hdf5" -> 0)
            basename = os.path.basename(file_path)
            episode_id = int(basename.split("_")[1].split(".")[0])
            expected_length = self.LENGTH_MAP[episode_id]

            with h5py.File(file_path, "r") as f:
                # Dataset existence
                self.assertIn("/value_assignment/rew_total", f)
                self.assertIn("/value_assignment/v_target", f)

                # Length matching (hardcoded)
                self.assertEqual(
                    len(f["/value_assignment/rew_total"][:]),
                    expected_length,
                    f"rew_total length mismatch for {basename}",
                )
                self.assertEqual(
                    len(f["/value_assignment/v_target"][:]),
                    expected_length,
                    f"v_target length mismatch for {basename}",
                )

    @patch("compute_value_targets.rewards.time_penalty")
    def test_run_overwrites_existing_datasets(self, mock_time_penalty):
        """Test that run overwrites existing value_assignment datasets."""
        # Track call count to differentiate between first run (ones) and second run (twos)
        call_count = 0

        def side_effect(qpos, actions, episode_id):
            nonlocal call_count
            call_count += 1
            # First 3 calls (first run): ones, next 3 calls (second run): twos
            value = 1.0 if call_count <= 3 else 2.0
            return np.full(len(qpos), value)

        mock_time_penalty.side_effect = side_effect

        computer = ComputeValueTargets.__new__(ComputeValueTargets)
        computer.cfg = self.test_config
        computer.gamma = 0.5

        # First run - should write values based on ones (-1.0 after weighting)
        computer.run()

        # Second run - should overwrite with values based on twos (-2.0 after weighting)
        computer.run()

        # Verify all three files have been overwritten with -2.0 values
        for file_path in self.test_files:
            basename = os.path.basename(file_path)
            episode_id = int(basename.split("_")[1].split(".")[0])
            expected_length = self.LENGTH_MAP[episode_id]
            expected_rew_total = np.full(expected_length, -2.0)

            with h5py.File(file_path, "r") as f:
                np.testing.assert_array_equal(
                    f["/value_assignment/rew_total"][:],
                    expected_rew_total,
                    f"File {basename} was not overwritten correctly",
                )

    @patch("compute_value_targets.rewards.time_penalty")
    def test_run_only_writes_specified_fields(self, mock_time_penalty):
        """Test that run only writes fields specified in step1_write_to_disk."""
        # Use side effect that adapts to each file's trajectory length
        mock_time_penalty.side_effect = lambda qpos, actions, episode_id: np.ones(
            len(qpos)
        )

        # Config that only writes rew_total (not v_target or individual components)
        config = self.test_config.copy()
        config["step1_write_to_disk"] = ["rew_total"]

        computer = ComputeValueTargets.__new__(ComputeValueTargets)
        computer.cfg = config
        computer.gamma = 0.99

        computer.run()

        # Verify only rew_total is written in all files
        for file_path in self.test_files:
            basename = os.path.basename(file_path)
            episode_id = int(basename.split("_")[1].split(".")[0])
            expected_length = self.LENGTH_MAP[episode_id]

            with h5py.File(file_path, "r") as f:
                self.assertIn("/value_assignment/rew_total", f)
                self.assertNotIn("/value_assignment/v_target", f)
                self.assertNotIn("/value_assignment/time_penalty", f)
                # Verify length matches expected trajectory length
                self.assertEqual(
                    len(f["/value_assignment/rew_total"][:]),
                    expected_length,
                    f"rew_total length mismatch for {basename}",
                )


if __name__ == "__main__":
    unittest.main()
