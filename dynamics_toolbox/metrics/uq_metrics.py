"""
Uncertainty quantification metrics.

Author: Ian Char
Date: April 17, 2023
"""
from typing import Tuple, Union

import numpy as np


def miscalibration_from_samples(
    samples: np.ndarray,
    truths: np.ndarray,
    include_overconfidence_scores: bool = False,
    fidelity: int = 99,
    use_intervals: bool = True,
) -> Union[np.ndarray, Tuple[np.ndarray]]:
    """Compute mean absolute miscalibration from samples for each predicted dimension.

    Args:
        samples: Samples from the model w shape (num_tests, num_samples, dim)
        truths: True points w shape (num_tests, dim)
        include_overconfidence_scores: Whether to compute overconfidence which is
            sum of the differences above 0.
        fidelity: Fidelity of the quantiles to use.
        use_intervals: Whether to measure from intervals or with quantiles.

    Returns: miscal and possibly overconfidence scores each w shape (dim,).
    """
    B, N, D = samples.shape
    miscals, overconfs = [], []
    exp_props = np.linspace(0.01, 0.99, 99)
    for d in range(D):
        if use_intervals:
            obs_props = np.array([
                np.mean(np.logical_and(
                    truths[:, d] >= np.quantile(samples[..., d], 0.5 - q / 2, axis=1),
                    truths[:, d] <= np.quantile(samples[..., d], 0.5 + q / 2, axis=1),
                ))
                for q in exp_props
            ])
        else:
            obs_props = np.array([
                np.mean(truths[:, d] <= np.quantile(samples[..., d], q, axis=1))
                for q in exp_props
            ])
        miscals.append(np.mean(np.abs(exp_props - obs_props)))
        overconfs.append(np.mean(np.maximum(exp_props - obs_props, 0)))
    if include_overconfidence_scores:
        return np.array(miscals), np.array(overconfs)
    return np.array(miscals)
