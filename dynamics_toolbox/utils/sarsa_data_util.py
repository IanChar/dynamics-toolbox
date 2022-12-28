"""
Utility for gym environments and gym data.

Author: Ian Char
"""
from typing import Dict, List, Optional, Sequence

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


def parse_into_snippet_datasets(
        qset: Dict[str, np.ndarray],
        snippet_size: Optional[int] = None,
        val_proportion: float = 0.0,
        test_proportion: float = 0.0,
        allow_padding: bool = False,
        shuffle: bool = True,
) -> List[Dict[str, np.ndarray]]:
    """Parse into a sarsa dataset into snippets of rollouts.

    Args:
        qset: Full dataset with multiple trajectories. The dataset at least has
            "observations", "actions", "next_observations", "rewards", "terminals".
        snippet_size: The length of the snippet size. If left as None, this will
            default to the smallest trajectory size in the dataset.
        val_proportion: Proportion of the trajectories to be used for validation.
        test_proportion: Proportion of the trajectories to be used for testing.
        allow_padding: Whether to allow padding of 0s if trajectory length is smaller
            than snippet_size.
        shuffle: Whether to shuffle the trajectories.

    Returns:
        Three dictionaries, one for each train, validation, and testing. Each contains
            * observations: ndarray of shape (num_snippets, snippet_size, obs_dim)
            * actions: ndarray of shape (num_snippets, snippet_size, act_dim)
            * next_observations: ndarray of shape (num_snippets, snippet_size, obs_dim)
            * rewards: ndarray of shape (num_snippets, snippet_size)
            * terminals: ndarray of shape (num_snippets, snippet_size)
            * mask: ndarray of 0s or 1s (num_snippets, snippet_size). This comes
                into play if allow_padding is True.
    """
    if val_proportion + test_proportion > 1:
        raise ValueError('Invalid validation and test proportions: '
                         f'{val_proportion} and {test_proportion}.')
    trajectories = parse_into_trajectories(qset)
    if shuffle:
        np.random.shuffle(trajectories)
    min_trajectory_length = np.min([len(traj['observations']) for traj in trajectories])
    if snippet_size is None:
        snippet_size = min_trajectory_length
    else:
        if snippet_size > min_trajectory_length and not allow_padding:
            raise ValueError(f'Cannot have snippet size ({snippet_size}) greater '
                             f'than min trajectory length ({min_trajectory_length}).')
    num_te = int(test_proportion * len(trajectories))
    te_trajectories, trajectories = trajectories[:num_te], trajectories[num_te:]
    num_val = int(val_proportion * len(trajectories))
    val_trajectories, trajectories = trajectories[:num_val], trajectories[num_val:]
    datasets = [{
        'observations': [],
        'actions': [],
        'next_observations': [],
        'rewards': [],
        'terminals': [],
        'mask': [],
    } for _ in range(3)]
    for dataset, trajectory_group in zip(datasets,
                                         [trajectories, val_trajectories,
                                          te_trajectories]):
        for traj in trajectory_group:
            traj_len = len(traj['observations'])
            if traj_len < snippet_size:
                for k, arr in traj.items():
                    if k in dataset:
                        arr_shape = list(arr.shape)
                        arr_shape[0] = snippet_size
                        padded = np.zeros(arr_shape)
                        padded[:traj_len] = arr
                        dataset[k].append(padded)
                dataset['mask'].append(np.array([1 if i < traj_len else 0
                                                 for i in range(snippet_size)]))
            else:
                for idx in range(traj_len + 1 - snippet_size):
                    for k, arr in dataset.items():
                        if k in traj:
                            arr.append(traj[k][idx:idx + snippet_size])
                    dataset['mask'].append(np.ones(snippet_size))
        for k, arr in dataset.items():
            dataset[k] = np.array(arr)
    return datasets
