"""
Data for loading and saving qdata to train on.

Author: Ian Char
"""
from typing import Dict

import gym
import h5py
import numpy as np


def get_qdata_from_data_source(data_source: str) -> Dict[str, np.ndarray]:
    """Get qdata from a data source.
    Args:
        data_source: Either the path to an h5py file or the name of a d4rl environment.
    Returns:
        Dictionary where keys include at least
            * "observations"
            * "actions"
            * "next_observations"
            * "rewards"
            * "terminals"
    """
    if '.h5py' in data_source:
        return load_qdata_from_hdf5(data_source)
    elif '-v' in data_source:
        return load_qdata_from_d4rl(data_source)
    else:
        raise ValueError(f'Did not recognize {data_source} as a path to .hdf5 file or environment.')


def load_qdata_from_hdf5(hdf5_path: str) -> Dict[str, np.ndarray]:
    """Get qdata from a a hdf5 file.
    Args:
        hdf5_path: Path to hdf5 file.
    Returns:
        Dictionary where keys include at least
            * "observations"
            * "actions"
            * "next_observations"
            * "rewards"
            * "terminals"
    """
    data = {}
    with h5py.File(hdf5_path, 'r') as hdata:
        for k, v in hdata.items():
            data[k] = v[()]
    return data


def load_qdata_from_d4rl(env_name: str) -> Dict[str, np.ndarray]:
    """Get qdata from d4rl.
    Args:
        env_name: Name of the environment.
    Returns:
        Dictionary where keys include at least
            * "observations"
            * "actions"
            * "next_observations"
            * "rewards"
            * "terminals"
    """
    # Lazy load so we don't require installation.
    import d4rl
    return d4rl.qlearning_dataset(gym.make(env_name))
