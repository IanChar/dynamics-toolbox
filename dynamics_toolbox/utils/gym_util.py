"""
Utility for gym environments and gym data.

Author: Ian Char
"""
from typing import Dict, List

import numpy as np


def parse_into_trajectories(
        dataset: Dict[str, np.ndarray],
) -> List[Dict[str, np.ndarray]]:
    """Parse a dataset into list of datasets separated by trajectory.

    Args:
        dataset: Full dataset with multiple trajectories. The dataset at least has
            "observations", "actions", "next_observations", "rewards", "terminals".

    Returns:
        List of datasets where each one is a trajectory.
    """
    trajectories = []
    start_idx = 0
    for end_idx in range(1, dataset['observations'].shape[0]):
        if end_idx == dataset['observations'].shape[0] - 1 or not np.allclose(
            dataset['next_observations'][end_idx - 1],
            dataset['observations'][end_idx],
        ):
            trajectories.append({k: v[start_idx:end_idx]
                                 for k, v in dataset.items()})
            start_idx = end_idx
    return trajectories

