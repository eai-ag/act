import numpy as np
import torch
import os
import h5py
from torch.utils.data import TensorDataset, DataLoader

import IPython

e = IPython.embed


class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.is_sim = None
        self.__getitem__(0)  # initialize self.is_sim

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        sample_full_episode = False  # hardcode

        episode_id = self.episode_ids[index]
        dataset_path = os.path.join(self.dataset_dir, f"episode_{episode_id}.hdf5")
        with h5py.File(dataset_path, "r") as root:
            is_sim = root.attrs["sim"]
            original_action_shape = root["/action"].shape
            episode_len = original_action_shape[0]
            if sample_full_episode:
                start_ts = 0
            else:
                start_ts = np.random.choice(episode_len)
            # get observation at start_ts only
            qpos = root["/observations/qpos"][start_ts]
            qvel = root["/observations/qvel"][start_ts]
            image_dict = dict()
            for cam_name in self.camera_names:
                image_dict[cam_name] = root[f"/observations/images/{cam_name}"][
                    start_ts
                ]
            # get all actions after and including start_ts
            if is_sim:
                action = root["/action"][start_ts:]
                action_len = episode_len - start_ts
            else:
                action = root["/action"][
                    max(0, start_ts - 1) :
                ]  # hack, to make timesteps more aligned
                action_len = episode_len - max(
                    0, start_ts - 1
                )  # hack, to make timesteps more aligned

        self.is_sim = is_sim
        padded_action = np.zeros(original_action_shape, dtype=np.float32)
        padded_action[:action_len] = action
        is_pad = np.zeros(episode_len)
        is_pad[action_len:] = 1

        # new axis for different cameras
        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)

        # construct observations
        image_data = torch.from_numpy(all_cam_images)
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        # channel last
        image_data = torch.einsum("k h w c -> k c h w", image_data)

        # normalize image and change dtype to float
        image_data = image_data / 255.0
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats[
            "action_std"
        ]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats[
            "qpos_std"
        ]

        return image_data, qpos_data, action_data, is_pad


def get_norm_stats(dataset_dir, num_episodes):
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
    # Concatenate instead of stack to handle variable lengths
    all_qpos_data = torch.cat(all_qpos_data, dim=0)
    all_action_data = torch.cat(all_action_data, dim=0)

    # normalize action data
    action_mean = all_action_data.mean(dim=0, keepdim=True)
    action_std = all_action_data.std(dim=0, keepdim=True)
    action_std = torch.clip(action_std, 1e-2, np.inf)  # clipping

    # normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=0, keepdim=True)
    qpos_std = all_qpos_data.std(dim=0, keepdim=True)
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf)  # clipping

    stats = {
        "action_mean": action_mean.numpy().squeeze().astype(np.float32),
        "action_std": action_std.numpy().squeeze().astype(np.float32),
        "qpos_mean": qpos_mean.numpy().squeeze().astype(np.float32),
        "qpos_std": qpos_std.numpy().squeeze().astype(np.float32),
        "example_qpos": qpos,
    }

    return stats


def collate_episodic(batch):
    images, qposes, actions, pads = zip(*batch)
    max_len = max(a.shape[0] for a in actions)
    padded_actions = [
        torch.cat([a, torch.zeros(max_len - a.shape[0], a.shape[1])], dim=0)
        for a in actions
    ]
    padded_pads = [
        torch.cat([p, torch.ones(max_len - p.shape[0], dtype=torch.bool)], dim=0) for p in pads
    ]
    return (
        torch.stack(images),
        torch.stack(qposes),
        torch.stack(padded_actions),
        torch.stack(padded_pads),
    )


def load_data(
    dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val
):
    print(f"\nData from: {dataset_dir}\n")
    # obtain train test split
    train_ratio = 0.8
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[: int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes) :]

    # obtain normalization stats for qpos and action
    norm_stats = get_norm_stats(dataset_dir, num_episodes)

    # construct dataset and dataloader
    train_dataset = EpisodicDataset(
        train_indices, dataset_dir, camera_names, norm_stats
    )
    val_dataset = EpisodicDataset(val_indices, dataset_dir, camera_names, norm_stats)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size_train,
        shuffle=True,
        pin_memory=True,
        num_workers=1,
        prefetch_factor=1,
        collate_fn=collate_episodic,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size_val,
        shuffle=True,
        pin_memory=True,
        num_workers=1,
        prefetch_factor=1,
        collate_fn=collate_episodic,
    )

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim


### env utils


def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])


def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose


### helper functions


def compute_dict_mean(epoch_dicts):
    first_dict = epoch_dicts[0]
    if 'num_valid' in first_dict:
        # Weighted average using 'num_valid' as weights
        total_weight = sum(d['num_valid'] for d in epoch_dicts)
        assert total_weight > 0, "All batches have 0 valid elements; check data integrity"
        result = {}
        for k in first_dict:
            if k == 'num_valid':
                continue
            weighted_sum = sum(d[k] * d['num_valid'] for d in epoch_dicts)
            result[k] = weighted_sum / total_weight
        return result
    else:
        # Equal-weight average (fallback for CNNMLP/other)
        result = {k: None for k in first_dict}
        num_items = len(epoch_dicts)
        for k in result:
            value_sum = 0
            for epoch_dict in epoch_dicts:
                value_sum += epoch_dict[k]
            result[k] = value_sum / num_items
        return result


def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        if k != 'num_valid':
            new_d[k] = v.detach()
        else:
            new_d[k] = v
    return new_d


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


class TemporalEnsembler:
    def __init__(
        self,
        chunk_size,
        action_dim,
        decay_rate=0.01,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        """
        chunk_size (k): The number of future actions predicted per query.
                         This also defines the history window length.
        """
        self.k = chunk_size
        self.action_dim = action_dim

        # Precompute weights: Newer chunks have higher weight
        # time_steps_since_prediction: 0 (newest) -> high weight, k-1 (oldest) -> low weight
        time_steps_since_prediction = torch.arange(self.k, device=device)
        weights = torch.exp(-decay_rate * time_steps_since_prediction)

        # Normalize weights so they sum to 1.0 for a true weighted average
        self.normalized_weights = (weights / weights.sum()).unsqueeze(-1)  # (k, 1)

        # This buffer stores the 'k' most recent predicted chunks.
        # Shape: (number_of_chunks, steps_per_chunk, action_dimensions)
        self.chunk_buffer = torch.zeros(
            (self.k, self.k, self.action_dim), device=device
        )

        self.current_ptr = 0  # Tracks which slot in the buffer to overwrite

    def get_ensembled_action(self, predicted_chunk):
        """
        predicted_chunk: The (k, action_dim) output from the Transformer at the current time.
        """
        # 1. Store the newest prediction in our circular buffer
        self.chunk_buffer[self.current_ptr] = predicted_chunk

        # 2. Identify the indices for the "Vertical Slice"
        # We need the 1st action from the newest chunk,
        # the 2nd action from the chunk predicted 1 step ago, etc.

        # 'chunk_indices' looks back in time: [current, current-1, current-2, ...]
        chunk_indices = (self.current_ptr - torch.arange(self.k)) % self.k

        # 'action_indices' looks forward in the chunk: [0, 1, 2, ..., k-1]
        action_indices = torch.arange(self.k)

        # 3. Extract the window of relevant actions
        # This gathers one action from each of the 'k' chunks in the buffer.
        action_window = self.chunk_buffer[
            chunk_indices, action_indices
        ]  # (k, action_dim)

        # 4. Compute the final action via weighted average
        # Higher weight is given to the action from the most recent chunk (index 0)
        final_action = torch.sum(action_window * self.normalized_weights, dim=0)

        # 5. Advance the pointer for the next time step
        self.current_ptr = (self.current_ptr + 1) % self.k

        return final_action
