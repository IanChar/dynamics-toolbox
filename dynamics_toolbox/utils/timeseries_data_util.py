"""
Utility for timeseries data which have only observations and next observations.

Author: Youngseog Chung
"""
from typing import Dict, List, Optional, Sequence

import numpy as np


def parse_into_separate_timeseries(
        dataset: Dict[str, np.ndarray],
) -> List[Dict[str, np.ndarray]]:
    """Parse a dataset into list of datasets separated by trajectory.

    Args:
        dataset: Full dataset with multiple trajectories. The dataset at least has
            "observations" and "next_observations".

    Returns:
        List of datasets where each one is a trajectory.
    """
    timeseries = []
    start_idx = 0
    for end_idx in range(1, dataset['observations'].shape[0]):
        if end_idx == dataset['observations'].shape[0] - 1 or not np.allclose(
                dataset['next_observations'][end_idx - 1],
                dataset['observations'][end_idx],
        ):
            timeseries.append({k: v[start_idx:end_idx]
                               for k, v in dataset.items()})
            start_idx = end_idx
    return timeseries


def split_single_timeseries_separate_timeseries(
        dataset: Dict[str, np.ndarray],
        select_idx: np.ndarray,
) -> List[Dict[str, np.ndarray]]:
    """Split a single timeseries into two separate timeseries, based on select_idx.

    Args:
        dataset: Full dataset with multiple trajectories. The dataset at least has
            "observations" and "next_observations".
        select_idx: Indices of the selected timeseries, must be a flat array of element
            indices, e.g. np.arange(10).

    Returns:
        The selected timeseries and the non-selected timeseries.
    """

    selected_timeseries = [
        {k: v[select_idx] for k, v in dataset.items()}
    ]
    non_seleted_idx = np.setdiff1d(np.arange(dataset['observations'].shape[0]),
                                   select_idx)
    non_selected_timeseries = [
        {k: v[non_seleted_idx] for k, v in dataset.items()}
    ]
    return selected_timeseries, non_selected_timeseries


def parse_into_timeseries_snippet_datasets(
        qset: Dict[str, np.ndarray],
        is_single_timeseries: bool,
        snippet_size: Optional[int] = None,
        val_proportion: float = 0.0,
        test_proportion: float = 0.0,
        allow_padding: bool = False,
        shuffle: bool = True,
) -> List[Dict[str, np.ndarray]]:
    """Parse into a sarsa dataset into snippets of rollouts.
        Parse a timeseries dataset into snippets of the timeseries.


    Args:
        qset: Full dataset with possibly multiple concatenated timeseries. The dataset
            at least has "observations".
        is_single_timeseries: True means qset consists of a single data-stream,
            instead of e.g. concatenated trajectories.
            If False, qset must have "next_observations" since it's impossible to know
            the next observation of disparate trajectories otherwise.
        snippet_size: The length of the snippet size (i.e. sequence length).
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

    if not is_single_timeseries:
        assert "next_observations" in qset
        timeseries = parse_into_separate_timeseries(qset)
    else:
        # create "next_observations" if provided data only has "observations"
        if "next_observations" not in qset:
            qset["next_observations"] = qset["observations"].copy()[1:]
            qset["observations"] = qset["observations"].copy()[:-1]
            assert qset["observations"].shape == qset["next_observations"].shape
        timeseries = [qset]

    if shuffle:
        np.random.shuffle(timeseries)

    min_ts_length = np.min([len(ts['observations']) for ts in timeseries])
    max_ts_length = np.max([len(ts['observations']) for ts in timeseries])
    avg_ts_length = np.mean([len(ts['observations']) for ts in timeseries])
    if snippet_size is None:
        # design choice of what the default snippet_size should be: min, max, or avg
        # snippet_size = avg_ts_length
        snippet_size = max_ts_length
    else:
        if snippet_size > min_ts_length and not allow_padding:
            raise ValueError(f'Cannot have snippet size ({snippet_size}) greater '
                             f'than min trajectory length ({min_ts_length}).')

    if is_single_timeseries:
        # if it is a single timeseries, split val and test from the single timeseries
        assert len(timeseries) == 1
        num_te = int(test_proportion * timeseries[0]["observations"].shape[0])
        num_val = int(val_proportion * timeseries[0]["observations"].shape[0])
        te_timeseries, timeseries = split_single_timeseries_separate_timeseries(
            timeseries[0], np.arange(num_te))
        val_timeseries, timeseries = split_single_timeseries_separate_timeseries(
            timeseries[0], np.arange(num_val))
    else:
        # if it is a multiple timeseries, split val and test with disparate timeseries
        num_te = int(test_proportion * len(timeseries))
        num_val = int(val_proportion * len(timeseries))
        te_timeseries, timeseries = timeseries[:num_te], timeseries[num_te:]
        val_timeseries, timeseries = timeseries[:num_val], timeseries[num_val:]

    datasets = [{
        'observations': [],
        'next_observations': [],
        'mask': [],
    } for _ in range(3)]
    for dataset, timeseries_group in zip(datasets,
                                         [timeseries, val_timeseries, te_timeseries]):
        for ts in timeseries_group:
            ts_len = len(ts['observations'])
            if ts_len < snippet_size:
                for k, arr in ts.items():
                    if k in dataset:
                        arr_shape = list(arr.shape)
                        arr_shape[0] = snippet_size
                        padded = np.zeros(arr_shape)
                        padded[:ts_len] = arr
                        dataset[k].append(padded)
                dataset['mask'].append(np.array([1 if i < ts_len else 0
                                                 for i in range(snippet_size)]))
            else:
                for idx in range(ts_len + 1 - snippet_size):
                    for k, arr in dataset.items():
                        if k in ts:
                            arr.append(ts[k][idx:idx + snippet_size])
                    dataset['mask'].append(np.ones(snippet_size))
        for k, arr in dataset.items():
            dataset[k] = np.array(arr)
    return datasets
