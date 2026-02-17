import numpy as np


def time_penalty(qpos, actions, episode_id):
    """
    Assigns 1 for each timestep to encourage faster task completion.

    Args:
        qpos: Array of joint positions, shape (T, qpos_dim)
        actions: Array of actions, shape (T, action_dim)
        episode_id: Episode identifier

    Returns:
        Array of rewards, shape (T,), with 1 at each timestep
    """
    return np.ones(len(qpos))
