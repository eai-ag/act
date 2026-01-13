import unittest
import sys
sys.path.append('/home/eai/act')  # Adjust path if needed
from utils import compute_dict_mean


class TestComputeDictMean(unittest.TestCase):
    def test_weighted_average(self):
        """Test weighted average with 'num_valid' (simulates ACT)."""
        epoch_dicts = [
            {'loss': 2.0, 'l1': 1.5, 'num_valid': 10},
            {'loss': 4.0, 'l1': 3.0, 'num_valid': 20},
        ]
        result = compute_dict_mean(epoch_dicts)
        # Total weight = 30, loss = (2*10 + 4*20)/30 = 3.333..., l1 = (1.5*10 + 3*20)/30 = 2.5
        self.assertAlmostEqual(result['loss'], 100/30, places=5)
        self.assertEqual(result['l1'], 2.5)

    def test_equal_weight_average(self):
        """Test equal-weight average without 'num_valid' (simulates CNNMLP)."""
        epoch_dicts = [
            {'loss': 1.0, 'l1': 0.5},
            {'loss': 3.0, 'l1': 2.5},
        ]
        result = compute_dict_mean(epoch_dicts)
        # 2 batches, loss = (1+3)/2 = 2.0, l1 = (0.5+2.5)/2 = 1.5
        self.assertEqual(result['loss'], 2.0)
        self.assertEqual(result['l1'], 1.5)

    def test_zero_num_valid_raises_error(self):
        """Test that all-zero 'num_valid' raises AssertionError."""
        epoch_dicts = [
            {'loss': 1.0, 'l1': 0.5, 'num_valid': 0},
            {'loss': 3.0, 'l1': 2.5, 'num_valid': 0},
        ]
        with self.assertRaises(AssertionError):
            compute_dict_mean(epoch_dicts)

    def test_weighted_vs_original(self):
        """Certify that weighted average differs from original equal-weight."""
        def compute_dict_mean_original(epoch_dicts):
            result = {k: None for k in epoch_dicts[0]}
            num_items = len(epoch_dicts)
            for k in result:
                value_sum = 0
                for epoch_dict in epoch_dicts:
                    value_sum += epoch_dict[k]
                result[k] = value_sum / num_items
            return result

        epoch_dicts = [
            {'loss': 2.0, 'l1': 1.5, 'num_valid': 10},
            {'loss': 4.0, 'l1': 3.0, 'num_valid': 20},
        ]
        result_new = compute_dict_mean(epoch_dicts)
        result_original = compute_dict_mean_original(epoch_dicts)
        # New: loss ≈ 3.333, original: 3.0
        self.assertGreater(result_new['loss'], result_original['loss'])
        self.assertNotAlmostEqual(result_new['loss'], result_original['loss'], delta=0.1)


if __name__ == "__main__":
    unittest.main()