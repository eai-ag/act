import unittest
import torch
import numpy as np
import sys

sys.path.insert(0, "/home/eai/act")
from utils import TemporalEnsembler


class TestTemporalEnsembler(unittest.TestCase):

    def setUp(self):
        self.num_queries = 5
        self.action_dim = 7
        self.num_timesteps = 20
        self.max_timesteps = self.num_timesteps
        self.state_dim = self.action_dim
        self.query_frequency = 1
        self.temporal_agg = True
        self.k = 0.01

    def _run_test_for_device(self, device):
        # Generate artificial chunks
        chunks = [
            torch.randn(self.num_queries, self.action_dim, device=device)
            for _ in range(self.num_timesteps)
        ]

        # Ensembler method
        ensembler = TemporalEnsembler(
            chunk_size=self.num_queries, action_dim=self.action_dim, device=device
        )
        ensembler_actions = []
        for chunk in chunks:
            action = ensembler.get_ensembled_action(chunk)
            ensembler_actions.append(action)

        # Eval method (exact copy-paste adapted)
        all_time_actions = torch.zeros(
            [self.max_timesteps, self.max_timesteps + self.num_queries, self.state_dim],
            device=device,
        )

        eval_actions = []
        for t in range(self.num_timesteps):
            # Simulate policy call
            all_actions = chunks[
                t
            ]  # (num_queries, action_dim), but in eval it's (1, num_queries, action_dim), but since squeeze later, ok
            if self.temporal_agg:
                all_time_actions[[t], t : t + self.num_queries] = all_actions
                actions_for_curr_step = all_time_actions[:, t]
                actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                actions_for_curr_step = actions_for_curr_step[actions_populated]
                exp_weights = np.exp(-self.k * np.arange(len(actions_for_curr_step)))
                exp_weights = exp_weights / exp_weights.sum()
                exp_weights = (
                    torch.from_numpy(exp_weights).float().to(device).unsqueeze(dim=1)
                )
                exp_weights = exp_weights.flip(
                    0
                )  # Reverse to give highest weight to newest, it's a bug in the original code
                raw_action = (actions_for_curr_step * exp_weights).sum(
                    dim=0, keepdim=True
                )
                eval_actions.append(raw_action.squeeze(0))

        # Assert match after buffer fills (t >= num_queries - 1)
        for t in range(self.num_queries - 1, self.num_timesteps):
            self.assertTrue(
                torch.allclose(
                    ensembler_actions[t], eval_actions[t], rtol=1e-5, atol=1e-6
                ),
                f"Mismatch at t={t}: ensembler={ensembler_actions[t]}, eval={eval_actions[t]}",
            )

    def test_cpu(self):
        device = torch.device("cpu")
        self._run_test_for_device(device)

    def test_cuda(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        device = torch.device("cuda")
        self._run_test_for_device(device)

    def test_weights_prioritize_newer_actions(self):
        """Test that TemporalEnsembler gives higher weight to newer actions."""
        device = torch.device("cpu")
        chunk_size = 3
        action_dim = 1
        decay_rate = 0.01  # Match class default
        ensembler = TemporalEnsembler(
            chunk_size=chunk_size,
            action_dim=action_dim,
            decay_rate=decay_rate,
            device=device,
        )

        # Check normalized_weights: exponentially decaying (newest > older)
        weights = ensembler.normalized_weights.squeeze()  # (chunk_size,)
        print(
            f"TemporalEnsembler weights (newest to oldest, exp decay k={decay_rate}): {[f'{w:.3f}' for w in weights.tolist()]}"
        )
        self.assertGreater(
            weights[0].item(),
            weights[1].item(),
            f"Newest weight {weights[0].item():.3f} should > next {weights[1].item():.3f}",
        )
        self.assertGreater(
            weights[1].item(),
            weights[2].item(),
            f"Middle weight {weights[1].item():.3f} should > oldest {weights[2].item():.3f}",
        )
        print("✓ TemporalEnsembler prioritizes newer actions (weights decrease)")

        # Hardcoded actions: use varied chunks
        chunkA = torch.tensor([[0.1], [0.2], [0.3]], device=device)
        chunkB = torch.tensor([[1.1], [1.2], [1.3]], device=device)
        chunkC = torch.tensor([[2.1], [2.2], [2.3]], device=device)
        chunkD = torch.tensor([[3.1], [3.2], [3.3]], device=device)

        # Fill buffer
        _ = ensembler.get_ensembled_action(chunkA)
        _ = ensembler.get_ensembled_action(chunkB)
        _ = ensembler.get_ensembled_action(chunkC)

        # Expected action_window after feeding chunkD: [3.1, 2.2, 1.3]
        # Extracted as: chunkD[0], chunkC[1], chunkB[2]
        action_window = torch.tensor([3.1, 2.2, 1.3], device=device)
        expected_sum = (weights * action_window).sum().item()
        products = (weights * action_window).tolist()
        print(
            f"Hardcoded action_window (newest to oldest): {[f'{a:.1f}' for a in action_window.tolist()]}"
        )
        print(f"Products (weights * actions): {[f'{p:.3f}' for p in products]}")
        print(f"Expected ensembled action: {expected_sum:.3f}")

        # Get actual ensembled action from feeding chunkD
        actual = ensembler.get_ensembled_action(chunkD).item()
        self.assertAlmostEqual(
            actual,
            expected_sum,
            places=5,
            msg=f"Actual {actual:.3f} should match expected {expected_sum:.3f}",
        )
        print(f"✓ Actual ensembled action {actual:.3f} matches expected")

    def test_original_eval_prioritizes_older_actions(self):
        """Test that original eval logic gives higher weight to older actions using exact eval code."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        temporal_agg = True
        num_queries = 3
        state_dim = 1
        max_timesteps = 3
        query_frequency = 1  # Since temporal_agg

        # Hardcoded actions: simulate policy outputs
        all_actions_t0 = torch.tensor(
            [[0.1], [0.2], [0.3]], device=device
        )  # Chunk at t=0
        all_actions_t1 = torch.tensor(
            [[1.1], [1.2], [1.3]], device=device
        )  # Chunk at t=1
        all_actions_t2 = torch.tensor(
            [[2.1], [2.2], [2.3]], device=device
        )  # Chunk at t=2

        # Initialize all_time_actions as in eval (exact code)
        all_time_actions = torch.zeros(
            [max_timesteps, max_timesteps + num_queries, state_dim], device=device
        )

        raw_action = None
        # Simulate the eval loop for t=0,1,2 (exact code from imitate_episodes.py)
        for t in range(max_timesteps):
            if t % query_frequency == 0:
                if t == 0:
                    all_actions = all_actions_t0
                elif t == 1:
                    all_actions = all_actions_t1
                elif t == 2:
                    all_actions = all_actions_t2
            if temporal_agg:
                all_time_actions[[t], t : t + num_queries] = all_actions
                actions_for_curr_step = all_time_actions[:, t]
                actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                actions_for_curr_step = actions_for_curr_step[actions_populated]
                k = 0.01
                exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                exp_weights = exp_weights / exp_weights.sum()
                exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                raw_action = (actions_for_curr_step * exp_weights).sum(
                    dim=0, keepdim=True
                )
                print()
            else:
                raw_action = all_actions[:, t % query_frequency]

        # For t=2, raw_action should be computed
        # Manually compute expected for t=2: actions_for_curr_step = [0.3, 1.2, 2.1]
        actions_for_curr_step_expected = torch.tensor(
            [[0.3], [1.2], [2.1]], device=device
        )
        exp_weights_expected = np.exp(-0.01 * np.arange(3))
        exp_weights_expected = exp_weights_expected / exp_weights_expected.sum()
        exp_weights_expected = (
            torch.from_numpy(exp_weights_expected).to(device).unsqueeze(dim=1)
        )
        expected_raw_action = (
            actions_for_curr_step_expected * exp_weights_expected
        ).sum(dim=0, keepdim=True)

        print(
            f"Expected actions_for_curr_step: {[f'{a:.1f}' for a in actions_for_curr_step_expected.squeeze().tolist()]}"
        )
        print(
            f"Expected weights: {[f'{w:.3f}' for w in exp_weights_expected.squeeze().tolist()]}"
        )
        products = (
            exp_weights_expected.squeeze() * actions_for_curr_step_expected.squeeze()
        ).tolist()
        print(f"Products (weights * actions): {[f'{p:.3f}' for p in products]}")
        print(f"Expected raw_action: {expected_raw_action.squeeze().item():.3f}")

        # Assert the computed raw_action matches expected
        self.assertAlmostEqual(
            raw_action.squeeze().item(),
            expected_raw_action.squeeze().item(),
            places=5,
            msg=f"Raw action {raw_action.squeeze().item():.3f} should match expected {expected_raw_action.squeeze().item():.3f}",
        )
        print(f"✓ Raw action from eval code matches expected")


if __name__ == "__main__":
    unittest.main()
