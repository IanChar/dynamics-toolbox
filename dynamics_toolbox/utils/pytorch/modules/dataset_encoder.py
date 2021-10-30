"""
Module for doing self attention encodings.

Author: Ian Char
Date: 10/26/2021
"""
import abc

import hydra.utils
import torch
from omegaconf import DictConfig


class DatasetEncoder(torch.nn.Module):

    @abc.abstractmethod
    def encode_dataset(self, net_in: torch.Tensor) -> torch.Tensor:
        """Make encodings for each input using attention.

        Args:
            net_in: The input with shape (num_input_groups, num_inputs, input_dim).

        Returns:
            An encoding with shape (num_input_groups, encoding_dim)
        """


class MLPDatasetEncoder(DatasetEncoder):

    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            mlp_cfg: DictConfig,
            **kwargs):
        """Constructor.

        Args:
            input_dim: The input dimension of each data point.
            output_dim: The dimension of the output encoding.
            mlp_cfg: The configuration for the MLP to do the encoding.
        """
        super().__init__()
        self._encode_net = hydra.utils.instantiate(
            mlp_cfg,
            input_dim=input_dim,
            output_dim=output_dim,
            _recursive_=False,
        )

    def encode_dataset(self, net_in: torch.Tensor) -> torch.Tensor:
        """Make encodings for each input using attention.

        Args:
            net_in: The input with shape (num_input_groups, num_inputs, input_dim).

        Returns:
            An encoding with shape (num_input_groups, encoding_dim)
        """
        return self._encode_net.forward(net_in).mean(dim=1)


class SelfAttentionDatasetEncoder(DatasetEncoder):
    """Takes several inputs and creates an encoding corresponding to these inputs."""

    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            query_net_cfg: DictConfig,
            key_net_cfg: DictConfig,
            value_net_cfg: DictConfig,
            **kwargs
    ):
        """Constructor.

        Args:
            input_dim: The input dimension of each data point.
            output_dim: The dimension of the output encoding.
            query_net_cfg: Configuration for an MLP for the query network.
            key_net_cfg: Configuration for an MLP for the key network.
            value_net_cfg: Configuration for an MLP for the value network.
        """
        super().__init__()
        self._query_net = hydra.utils.instantiate(
            query_net_cfg,
            input_dim=input_dim,
            output_dim=output_dim,
            _recursive_=False,
        )
        self._key_net = hydra.utils.instantiate(
            key_net_cfg,
            input_dim=input_dim,
            output_dim=output_dim,
            _recursive_=False,
        )
        self._value_net = hydra.utils.instantiate(
            value_net_cfg,
            input_dim=input_dim,
            output_dim=output_dim,
            _recursive_=False,
        )

    def encode_dataset(self, net_in: torch.Tensor) -> torch.Tensor:
        """Make encodings for each input using attention.

            Args:
                net_in: The input with shape (num_input_groups, num_inputs, input_dim).

            Returns:
                An encoding with shape (num_input_groups, encoding_dim)
            """
        queries = self._query_net(net_in)
        keys = self._key_net(net_in)
        values = self._value_net(net_in)
        scores = torch.stack([queries[i] @ keys[i].T for i in range(len(queries))])
        scores = torch.nn.functional.softmax(scores, dim=-1)
        weighted_values = values.unsqueeze(-2) * scores.unsqueeze(-1)
        return weighted_values.sum(dim=2).mean(dim=1)
