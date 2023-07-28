"""
Uncertainty quantification metrics.

Author: Ian Char
Date: April 17, 2023
"""
from typing import Tuple, Union

import numpy as np
from scipy.stats import chi2


def multivariate_elipsoid_miscalibration(
    means: np.ndarray,
    stds: np.ndarray,
    truths: np.ndarray,
    include_overconfidence_scores: bool = False,
    fidelity: int = 99,
) -> float:
    """This is multi-dimensional miscalibration for the case in which each dimension
       is an independent Gaussian. When this happens the normalized residuals form
       a chi-squared distribution. We can then form elipsoids corresponding to some
       coverage and see how many points for in that.

    Args:
        means: Mean estimate w shape (num_samples, num_dims)
        stds: Standard estimate w shape (num_samples, num_dims)
        truths: The true observations w shape (num_samples, num_dims)
        include_overconfidence_scores: Whether to compute overconfidence which is
            sum of the differences above 0.
        fidelity: How many coverages to check.

    Returns: Single miscalibration score.
    """
    N, D = means.shape
    # Form the square residuals.
    normd_resids = np.sum(np.square((means - truths) / stds), axis=-1)
    # Find the thresholds for the residuals.
    exp_props = np.linspace(0.05, 0.95, fidelity)
    thresholds = chi2(D).ppf(exp_props)
    obs_props = np.mean(normd_resids <= thresholds.reshape(-1, 1), axis=1)
    miscal = np.mean(np.abs(exp_props - obs_props))
    if include_overconfidence_scores:
        return miscal, np.mean(np.maximum(exp_props - obs_props, 0))
    return miscal


def miscalibration_from_samples(
    samples: np.ndarray,
    truths: np.ndarray,
    fidelity: int = 99,
    use_intervals: bool = True,
) -> Union[np.ndarray, Tuple[np.ndarray]]:
    """Compute mean absolute miscalibration from samples for each predicted dimension.

    Args:
        samples: Samples from the model w shape (num_tests, num_samples, dim)
        truths: True points w shape (num_tests, dim)
        fidelity: Fidelity of the quantiles to use.
        use_intervals: Whether to measure from intervals or with quantiles.

    Returns:
        * Miscal (dim,)
        * Overconfidence (dim,)
        * Obs props (dim, fidelity)
        * Exp props (dim, fidelity)
    miscal and possibly overconfidence scores each w shape (dim,).
    """
    B, N, D = samples.shape
    miscals, overconfs, ops, eps = [], [], [], []
    exp_props = np.linspace(0, 1, fidelity + 2)[1:-1]
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
        ops.append(obs_props)
        eps.append(exp_props)
    return np.array(miscals), np.array(overconfs), np.array(ops), np.array(eps)


def coverage_and_sharpness_from_samples(
    samples: np.ndarray,
    truths: np.ndarray,
    coverage_amount: float,
) -> Tuple[np.ndarray]:
    """Calculate the amount the that is covered.

    Args:
        samples: Samples from the model w shape (num_tests, num_samples, dim)
        truths: True points w shape (num_tests, dim)
        coverage_amount: Target coverage.

    Returns:
        * Coverage for D dimensional box made from predictions.
        * Coverage per dimension (dim,)
        * Sharpness of each dimension (dim,)
    """
    bounds = [(np.quantile(samples[..., d], 0.5 - coverage_amount / 2, axis=1),
               np.quantile(samples[..., d], 0.5 + coverage_amount / 2, axis=1))
              for d in range(samples.shape[-1])]
    sharpnesses = np.array([np.mean(b[1] - b[0]) for b in bounds])
    covered = np.array([
        np.logical_and(truths[..., d] >= bounds[d][0], truths[..., d] <= bounds[d][1])
        for d in range(samples.shape[-1])
    ])
    coverages = np.mean(covered, axis=1)
    return np.mean(np.prod(covered, axis=0)), coverages, sharpnesses
