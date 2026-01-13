import unittest
import torch
import h5py
from torch.utils.data import DataLoader
from utils import EpisodicDataset, collate_episodic, get_norm_stats
from constants import SIM_TASK_CONFIGS

class TestCollateReal(unittest.TestCase):
    def test_collate_with_real_data(self):
        # Use real fruits_picking data, override incorrect dataset_dir in constants
        task_config = SIM_TASK_CONFIGS['fruits_picking'].copy()
        task_config['dataset_dir'] = 'data/fruits_picking'  # Correct path
        
        # Compute norm stats from real data
        norm_stats = get_norm_stats(task_config['dataset_dir'], task_config['num_episodes'])
        
        # Create dataset with first 2 episodes for small batch
        episode_ids = [0, 1]
        dataset = EpisodicDataset(episode_ids, task_config['dataset_dir'], task_config['camera_names'], norm_stats)
        
        # DataLoader with collate_episodic
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_episodic)
        
        # Get one batch
        batch = next(iter(dataloader))
        images, qposes, actions, pads = batch
        
        # Assertions
        batch_size = 2
        num_cams = len(task_config['camera_names'])  # 2: gripper, main
        action_dim = 7  # From STATE_DIM
        qpos_dim = 7   # From STATE_DIM
        
        # Shapes: images (batch, cams, C, H, W) - C=3, H=480, W=640 from typical
        self.assertEqual(images.shape[0], batch_size)
        self.assertEqual(images.shape[1], num_cams)
        self.assertEqual(images.shape[2], 3)  # RGB
        self.assertEqual(images.shape[3], 480)  # Height
        self.assertEqual(images.shape[4], 640)  # Width
        
        # qposes (batch, qpos_dim)
        self.assertEqual(qposes.shape, (batch_size, qpos_dim))
        
        # actions (batch, max_T, action_dim) - max_T should be max of episode lengths
        self.assertEqual(actions.shape[0], batch_size)
        self.assertEqual(actions.shape[2], action_dim)
        max_T = actions.shape[1]
        
        # Check max_T is the max episode length in batch
        episode_lengths = []
        for ep_id in episode_ids:
            dataset_path = f"{task_config['dataset_dir']}/episode_{ep_id}.hdf5"
            with h5py.File(dataset_path, 'r') as f:
                ep_len = f['/action'].shape[0]
                episode_lengths.append(ep_len)
        expected_max_T = max(episode_lengths)
        self.assertEqual(max_T, expected_max_T)
        
        # pads (batch, max_T)
        self.assertEqual(pads.shape, (batch_size, max_T))
        print(f"Pads dtype: {pads.dtype}")
        self.assertTrue(pads.dtype == torch.bool)
        
        # Padding checks: pads should have 0s for valid, 1s for padded
        for i in range(batch_size):
            # At least some valid (not all padded)
            self.assertTrue(torch.any(pads[i] == 0))
        
        print(f"Batch shapes: images {images.shape}, qposes {qposes.shape}, actions {actions.shape}, pads {pads.shape}")
        print("Collate test with real data passed!")

if __name__ == '__main__':
    unittest.main()