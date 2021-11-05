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
            min_num_conditioning: int,
            max_num_conditioning: int,
            input_is_last_in_batch: bool = False,
            input_can_be_conditioned: bool = False,
    ):
        """Constructor.

        Args:
            min_num_conditioning: The minimum number of points to condition on.
            max_num_conditioning: The maximum number of points to condition on.
            input_is_last_in_batch: Whether the input point should always be the last
                point in the batch or whether it should be
            input_can_be_conditioned: Whether or not the input of the network can
                also be a point conditioned on.
        """
        self._min_num_conditioning = min_num_conditioning
        self._max_num_conditioning = max_num_conditioning
        self._input_is_last_in_batch = input_is_last_in_batch
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
                return torch.cat([xi, yi], dim=1), xi, yi
            return None, xi, yi
        pred_idx = xi.shape[1] - 1 if self._input_is_last_in_batch \
            else np.random.randint(xi.shape[1])
        pred_x, pred_y = xi[:, pred_idx, :], yi[:, pred_idx, :]
        num_conditions = np.random.randint(
            min(self._min_num_conditioning, xi.shape[1] - 1),
            min(self._max_num_conditioning, xi.shape[1] - 1)
        )
        if num_conditions == 0:
            return None, pred_x, pred_y
        condition_idxs = np.random.shuffle(
            [i for i in range(len(xi.shape[1])) if i != pred_idx]
        )[:num_conditions]
        conditions = torch.cat([xi[:, condition_idxs, :], yi[:, condition_idxs, :]],
                               dim=-1)
        return conditions, pred_x, pred_y


class HistoryConditionSampler(ConditionSampler):

    def __init__(
            self,
            min_num_conditioning: int,
            max_num_conditioning: int,
            input_is_last_in_batch: bool = False,
    ):
        """Constructor.

        Args:
            min_num_conditioning: The minimum number of points to condition on.
            max_num_conditioning: The maximum number of points to condition on.
            input_is_last_in_batch: Whether the input point should always be the last
                point in the batch or whether it should be
        """
        self._min_num_conditioning = min_num_conditioning
        self._max_num_conditioning = max_num_conditioning
        self._input_is_last_in_batch = input_is_last_in_batch

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
            min(self._min_num_conditioning, xi.shape[1] - 1),
            min(self._max_num_conditioning, xi.shape[1] - 1)
        )
        pred_idx = xi.shape[1] - 1 if self._input_is_last_in_batch \
            else np.random.randint(num_conditions, xi.shape[1])
        pred_x, pred_y = xi[:, pred_idx, :], yi[:, pred_idx, :]
        if num_conditions == 0:
            return None, pred_x, pred_y
        conditions = torch.cat([xi[:, pred_idx - num_conditions:pred_idx, :],
                                yi[:, pred_idx - num_conditions:pred_idx, :]],
                               dim=-1)
        return conditions, pred_x, pred_y


class SelfConditionSampler(ConditionSampler):

    def __init__(
            self,
            input_is_last_in_batch: bool = False,
    ):
        """Constructor.

        Args:
            input_is_last_in_batch: Whether the input point should always be the last
                point in the batch or whether it should be
        """
        self._input_is_last_in_batch = input_is_last_in_batch

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
            return torch.cat([xi, yi], dim=1), xi, yi
        pred_idx = xi.shape[1] - 1 if self._input_is_last_in_batch \
            else np.random.randint(xi.shape[1])
        pred_x, pred_y = xi[:, pred_idx, :], yi[:, pred_idx, :]
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
