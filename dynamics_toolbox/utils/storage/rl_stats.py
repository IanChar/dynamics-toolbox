"""
Utility functions to read in statistics.
"""
import os

import pandas as pd


def read_stats_into_df(path, allow_missing=False):
    """Assemble the statistics for rlkit or rnn runs.

    Args:
        path: The base path with all the seed runs.

    Returns: A dataframe with all of the statistics.
    """
    subdfs = []
    for seed in os.listdir(path):
        if 'DS_Store' not in seed and 'hydra' not in seed:
            progress_path = os.path.join(path, seed, 'stats.txt')
            try:
                curr_df = pd.read_csv(progress_path)
            except FileNotFoundError:
                if allow_missing:
                    continue
                else:
                    raise FileNotFoundError(f'Cannot find {progress_path}.')
            curr_df.insert(0, 'seed', [int(seed) for _ in range(len(curr_df))])
            subdfs.append(curr_df)
    return pd.concat(subdfs, ignore_index=True)
