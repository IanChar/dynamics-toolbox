"""
Custom metrics for evaluation.

Author: Ian Char
Date: February 3, 2023
"""
import abc
import torch


class SequenceMetrics(metaclass=abc.ABCMeta):

    def __call__(self, ypred: torch.Tensor, ytrue: torch.Tensor, mask: torch.Tensor):
        """Compute metric.

        Args:
            ypred: Predicted y value with shape (batch_size, seq_len, dim)
            ytru: True y value with shape (batch_size, seq_len, dim)
            mask: Masks with shape (batch_size, seq_len, 1)
        """


class SequentialExplainedVariance(SequenceMetrics):

    def __init__(
        self,
        compute_type: str = 'average',
    ):
        """Constructor.

        Args:
            compute_type: Type of computation to do. Either 'average' for averaging
                over the dimensions or 'raw_values' for each dimension separately.
        """
        assert compute_type in ('raw_values', 'average')
        self.compute_type = compute_type

    def __call__(self, ypred: torch.Tensor, ytrue: torch.Tensor, mask: torch.Tensor):
        """Compute metric.

        Args:
            ypred: Predicted y value with shape (batch_size, seq_len, dim)
            ytru: True y value with shape (batch_size, seq_len, dim)
            mask: Masks with shape (batch_size, seq_len, 1)
        """
        num_real = mask.sum()
        resids = ytrue - ypred
        mean_resids = resids.sum(dim=(0, 1)) / num_real
        var_resids = (resids - mean_resids).square().sum(dim=(0, 1)) / num_real
        mean_true = ytrue.sum(dim=(0, 1)) / num_real
        var_true = (ytrue - mean_true).square().sum(dim=(0, 1)) / num_real
        evs = 1 - var_resids / var_true
        if self.compute_type == 'average':
            return evs.mean()
        else:
            return evs
