"""
Class for picking out what should be the conditioned from a batch of sequential data.

Author: Ian Char
Date: 11/5/2021
"""
import abc
from typing import Tuple, Sequence, Union

import hydra.utils
import numpy as np
import torch
from omegaconf import DictConfig


class ConditionSampler(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def split_batch(self, batch: Sequence[torch.Tensor]) \
            -> Tuple[Union[torch.Tensor, None], torch.Tensor, torch.Tensor]:
        """Split a batch into conditions and inputs.

        Args:
            batch: Batch to be split.

        Returns:
            Tuple of (conditioning, network input, labels), conditioning could be
            None if there is no conditioning to be made. Conditioning is a tensor with
            shape (batch_size, num_conditions, condition_dim). The input is a tensor
            with size (batch_size, input_dim). Output is a tensor with shape
            (batch_size, output_dim).
        """


class RandomSubsetConditionSampler(ConditionSampler):

    def __init__(
            self,
            num_predictions: int,
            min_num_conditioning: int,
            max_num_conditioning: int,
            predictions_are_last: bool = False,
            input_can_be_conditioned: bool = False,
    ):
        """Constructor.

        Args:
            num_predictions: The number of prediction points to make per prediction.
            min_num_conditioning: The minimum number of points to condition on.
            max_num_conditioning: The maximum number of points to condition on.
            predictions_are_last: Whether the point(s) to predict point should always
                be the ones at the end of the trajectory.
            input_can_be_conditioned: Whether or not the input of the network can
                also be a point conditioned on.
        """
        self._num_predictions = num_predictions
        self._min_num_conditioning = min_num_conditioning
        self._max_num_conditioning = max_num_conditioning
        self._predictions_are_last = predictions_are_last
        self._input_can_be_conditioned = input_can_be_conditioned

    def split_batch(self, batch: Sequence[torch.Tensor]) \
            -> Tuple[Union[torch.Tensor, None], torch.Tensor, torch.Tensor]:
        """Split a batch into conditions and inputs.

        Args:
            batch: Batch to be split.

        Returns:
            Tuple of (conditioning, network input, labels), conditioning could be
            None if there is no conditioning to be made. Conditioning is a tensor with
            shape (batch_size, num_conditions, condition_dim). The input is a tensor
            with size (batch_size, input_dim). Output is a tensor with shape
            (batch_size, output_dim).
        """
        xi, yi = batch
        if len(xi.shape) == 2:
            if self._input_can_be_conditioned:
                return torch.cat([xi, yi], dim=1).unsqueeze(0), xi, yi
            return None, xi, yi
        idxs_to_choose = list(range(xi.shape[1]))
        if not self._predictions_are_last:
            np.random.shuffle(idxs_to_choose)
        pred_idxs = idxs_to_choose[-self._num_predictions:]
        pred_x = xi[:, pred_idxs, :].reshape(-1, xi.shape[-1])
        pred_y = yi[:, pred_idxs, :].reshape(-1, yi.shape[-1])
        if not self._input_can_be_conditioned:
            idxs_to_choose = idxs_to_choose[:-self._num_predictions]
        np.random.shuffle(idxs_to_choose)
        num_conditions = np.random.randint(
            min(self._min_num_conditioning, len(idxs_to_choose)),
            min(self._max_num_conditioning, len(idxs_to_choose)),
        )
        if num_conditions == 0:
            return None, pred_x, pred_y
        condition_idxs = idxs_to_choose[:num_conditions]
        conditions = torch.cat([xi[:, condition_idxs, :], yi[:, condition_idxs, :]],
                               dim=-1)
        conditions = conditions.repeat_interleave(self._num_predictions, dim=0)
        return conditions, pred_x, pred_y


class HistoryConditionSampler(ConditionSampler):

    def __init__(
            self,
            num_predictions: int,
            min_num_conditioning: int,
            max_num_conditioning: int,
            predictions_are_last: bool = False,
    ):
        """Constructor.

        Args:
            num_predictions: The number of prediction points to make per prediction.
            min_num_conditioning: The minimum number of points to condition on.
            max_num_conditioning: The maximum number of points to condition on.
            predictions_are_last: Whether the input point should always be the last
                point in the batch or whether it should be
        """
        self._num_predictions = num_predictions
        self._min_num_conditioning = min_num_conditioning
        self._max_num_conditioning = max_num_conditioning
        self._predictions_are_last = predictions_are_last

    def split_batch(self, batch: Sequence[torch.Tensor]) \
            -> Tuple[Union[torch.Tensor, None], torch.Tensor, torch.Tensor]:
        """Split a batch into conditions and inputs.

        Args:
            batch: Batch to be split.

        Returns:
            Tuple of (conditioning, network input, labels), conditioning could be
            None if there is no conditioning to be made. Conditioning is a tensor with
            shape (batch_size, num_conditions, condition_dim). The input is a tensor
            with size (batch_size, input_dim). Output is a tensor with shape
            (batch_size, output_dim).
        """
        xi, yi = batch
        if len(xi.shape) == 2:
            return None, xi, yi
        num_conditions = np.random.randint(
            min(self._min_num_conditioning, xi.shape[1] - self._num_predictions),
            min(self._max_num_conditioning, xi.shape[1] - self._num_predictions)
        )
        if self._predictions_are_last:
            split_idx = xi.shape[1] - self._num_predictions
        else:
            split_idx = np.random.randint(
                    num_conditions, xi.shape[1] + 1 - self._num_predictions)
        pred_idxs = np.arange(split_idx, split_idx + self._num_predictions)
        pred_x = xi[:, pred_idxs, :].reshape(-1, xi.shape[-1])
        pred_y = yi[:, pred_idxs, :].reshape(-1, yi.shape[-1])
        if num_conditions == 0:
            return None, pred_x, pred_y
        cond_idxs = np.arange(split_idx - num_conditions, split_idx)
        conditions = torch.cat([xi[:, cond_idxs, :], yi[:, cond_idxs, :]], dim=-1)
        conditions = conditions.repeat_interleave(self._num_predictions, dim=0)
        return conditions, pred_x, pred_y


class SelfConditionSampler(ConditionSampler):

    def __init__(
            self,
            predictions_are_last: bool = False,
    ):
        """Constructor.

        Args:
            predictions_are_last: Whether the input point should always be the last
                point in the batch or whether it should be
        """
        self._predictions_are_last = predictions_are_last

    def split_batch(self, batch: Sequence[torch.Tensor]) \
            -> Tuple[Union[torch.Tensor, None], torch.Tensor, torch.Tensor]:
        """Split a batch into conditions and inputs.

        Args:
            batch: Batch to be split.

        Returns:
            Tuple of (conditioning, network input, labels), conditioning could be
            None if there is no conditioning to be made. Conditioning is a tensor with
            shape (batch_size, num_conditions, condition_dim). The input is a tensor
            with size (batch_size, input_dim). Output is a tensor with shape
            (batch_size, output_dim).
        """
        xi, yi = batch
        if len(xi.shape) == 2:
            return torch.cat([xi, yi], dim=1).unsqueeze(0), xi, yi
        if self._predictions_ar_last:
            pred_idxs = np.arange(xi.shape[1] - self._num_predictions, xi.shape[1])
        else:
            pred_idxs = np.arange(xi.shape[1])
            np.random.shuffle(pred_idxs)
            pred_idxs = pred_idxs[:self._num_predictions]
        pred_x, pred_y = xi[:, pred_idxs, :], yi[:, pred_idxs, :]
        conditions = torch.cat([pred_x, pred_y], dim=1).unsqueeze(1)
        return conditions, pred_x, pred_y


class MixtureConditionSampler(ConditionSampler):

    def __init__(
            self,
            sampler_cfgs: Sequence[DictConfig],
            sampler_probabilities: Sequence[float],
    ):
        """Constructor.

        Args:
            sampler_cfgs: The configurations for the samplers.
            sampler_probabilities: Probability that a specific sampler will be chosen.
        """
        assert len(sampler_cfgs) == len(sampler_probabilities), \
            'Number of samplers must match number of probabilities.'
        for idx, cfg in enumerate(sampler_cfgs):
            setattr(self, f'_sampler_{idx}', hydra.utils.instantiate(cfg,
                                                                     _recursive_=False))
        self._probabilities = sampler_probabilities
        self._num_samplers = len(sampler_cfgs)

    def split_batch(self, batch: Sequence[torch.Tensor]) \
            -> Tuple[Union[torch.Tensor, None], torch.Tensor, torch.Tensor]:
        """Split a batch into conditions and inputs.

        Args:
            batch: Batch to be split.

        Returns:
            Tuple of (conditioning, network input, labels), conditioning could be
            None if there is no conditioning to be made. Conditioning is a tensor with
            shape (batch_size, num_conditions, condition_dim). The input is a tensor
            with size (batch_size, input_dim). Output is a tensor with shape
            (batch_size, output_dim).
        """
        sampler_idx = np.random.choice(self._num_samplers, p=self._probabilities)
        return getattr(self, f'_sampler_{sampler_idx}').split_batch(batch)
