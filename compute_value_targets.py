import h5py
import numpy as np
import os
import glob
import scipy.signal
import yaml
import rewards


def discount_cumsum(x, discount):
    """
    Magic from rllab for computing discounted cumulative sums of vectors.

    input:
        vector x, [x0, x1, x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]

    Reference:
    https://github.com/openai/spinningup/blob/038665d62d569055401d91856abb287263096178/spinup/algos/pytorch/ppo/core.py#L29
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class ComputeValueTargets:
    """Computes rewards and value targets for RL trajectories."""

    def __init__(self, config_path):
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)
        self.gamma = self.cfg["rl_params"]["gamma"]

    def compute_rewards(self, qpos, actions, episode_id):
        """
        Compute total reward for a trajectory using the reward pipeline.

        Args:
            qpos: Joint positions array
            actions: Actions array
            episode_id: Episode identifier

        Returns:
            dict: Dictionary containing individual reward components and total reward
        """
        reward_components = {}
        total_reward = np.zeros(len(qpos))

        for item in self.cfg["reward_pipeline"]:
            func_name = item["name"]
            weight = item["weight"]

            # Get reward function from rewards.py
            reward_func = getattr(rewards, func_name)
            component_reward = reward_func(
                qpos=qpos, actions=actions, episode_id=episode_id
            )

            # Store individual component
            reward_components[func_name] = component_reward

            # Add weighted component to total
            total_reward += component_reward * weight

        reward_components["rew_total"] = total_reward
        return reward_components

    def compute_value_targets(self, rewards):
        """
        Compute value targets (returns-to-go) using discounted cumulative sum.

        This computes: V_t = r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + ...

        Args:
            rewards: Array of rewards for the trajectory

        Returns:
            Array of value targets

        Reference:
        https://github.com/openai/spinningup/blob/038665d62d569055401d91856abb287263096178/spinup/algos/pytorch/ppo/ppo.py#L67
        """
        return discount_cumsum(rewards, self.gamma)

    def process_trajectory(self, qpos, actions, episode_id):
        """
        Process a single trajectory to compute rewards and value targets.

        Args:
            qpos: Joint positions array
            actions: Actions array
            episode_id: Episode identifier

        Returns:
            dict: Dictionary containing all computed values
        """
        # Compute all reward components
        results = self.compute_rewards(qpos, actions, episode_id)

        # Compute value targets from total reward
        results["v_target"] = self.compute_value_targets(results["rew_total"])

        return results

    def run(self):
        """Process all HDF5 files in the data directory."""
        data_dir = self.cfg["data_dir"]
        files = sorted(glob.glob(os.path.join(data_dir, "*.hdf5")))

        if not files:
            print(f"No HDF5 files found in {data_dir}")
            return

        print(f"Found {len(files)} HDF5 files to process")

        for f_path in files:
            # Extract episode_id from filename (e.g., episode_5.hdf5 -> 5)
            try:
                episode_id = int(os.path.basename(f_path).split("_")[1].split(".")[0])
            except (IndexError, ValueError):
                episode_id = 0
                print(
                    f"Warning: Could not extract episode_id from {os.path.basename(f_path)}, using 0"
                )

            with h5py.File(f_path, "a") as f:
                # Load trajectory data
                qpos = f["/observations/qpos"][:]
                actions = f["/action"][:]

                # Compute rewards and value targets
                results = self.process_trajectory(qpos, actions, episode_id)

                # Write specified fields to disk
                for field_name in self.cfg["step1_write_to_disk"]:
                    if field_name in results:
                        ds_path = f"/value_assignment/{field_name}"

                        # Delete existing dataset if present
                        if ds_path in f:
                            del f[ds_path]

                        # Create new dataset
                        f.create_dataset(
                            ds_path, data=results[field_name], compression="gzip"
                        )
                    else:
                        print(
                            f"Warning: {field_name} not found in results for {os.path.basename(f_path)}"
                        )

            print(f"✓ Processed {os.path.basename(f_path)} (episode {episode_id})")

        print(f"\nSuccessfully processed {len(files)} files")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default="rl_cfg.yaml",
        help="Path to config file (default: rl_config.yaml)",
    )
    args = parser.parse_args()

    computer = ComputeValueTargets(args.config)
    computer.run()
