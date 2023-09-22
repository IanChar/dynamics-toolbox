"""
A standard Q network.

Author: Ian Char
Date: April 6, 2023
"""
from typing import Callable, Optional, Sequence

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import functional as F

from dynamics_toolbox.rl.modules.history_encoders.abstract_history_encoder import (
    HistoryEncoder,
)
from dynamics_toolbox.utils.pytorch.modules.fc_network import FCNetwork


class QNet(FCNetwork):

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: Sequence[int],
        hidden_activation: Callable[[Tensor], Tensor] = F.relu,
        input_dim: Optional[int] = None,
        **kwargs
    ):
        """Constructor.

        obs_dim: Observation dimension size.
        act_dim: Action dimension size.
        hidden_sizes: Number of hidden units for each hidden layer.
        hidden_activation: Activation function.
        """
        super().__init__(
            input_dim=obs_dim + act_dim if input_dim is None else input_dim,
            output_dim=1,
            hidden_sizes=hidden_sizes,
            hidden_activation=hidden_activation,
        )
        self._obs_dim = obs_dim
        self._act_dim = act_dim

    def forward(self, obs: Tensor, act: Tensor) -> Tensor:
        """Forward pass.

        Args:
            obs: Observation of shape (batch_size, obs_dim)
            act: Action of shape (batch_size, obs_dim).

        Returns: Value of shape (batch_size, 1)
        """
        return super().forward(torch.cat([obs, act], dim=-1))


class SequentialQNet(FCNetwork):

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: Sequence[int],
        history_encoder: HistoryEncoder,
        obs_act_encode_dim: int,
        hidden_activation: Callable[[Tensor], Tensor] = F.relu,
        input_dim: Optional[int] = None,
        **kwargs
    ):
        """Constructor.

        Args:
            history_encoder: Encoder for the history.
            obs_act_encode_dim: The size of the encoding for joint obs, act encoding.
            All the rest are the same as QNet.
        """
        super().__init__(
            input_dim=(input_dim if input_dim is not None
                       else history_encoder.out_dim + obs_act_encode_dim),
            output_dim=1,
            hidden_sizes=hidden_sizes,
            hidden_activation=hidden_activation,
            **kwargs
        )
        self._obs_dim = obs_dim
        self._act_dim = act_dim
        self._history_encoder = history_encoder
        self._obs_act_encoder = nn.Linear(self._obs_dim + self._act_dim,
                                          obs_act_encode_dim)

    def forward(
        self,
        obs_seq: Tensor,
        prev_act_seq: Tensor,
        rew_seq: Tensor,
        act_seq: Tensor,
        encode_init: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass. Where B = batch_size, L = seq length..

        Args:
            obs_seq: (B, L, obs_dim)
            prev_act_seq: (B, L, act_dim)
            rew_seq: (B, L, 1)
            act_seq: (B, L, act_dim)

        Returns: Tensor
        """
        encoding = self._history_encoder(obs_seq, prev_act_seq, rew_seq,
                                         encode_init=encode_init)[0]
        obs_act_encoding = self.hidden_activation(self._obs_act_encoder(
            torch.cat([obs_seq, act_seq], dim=-1)))
        return super().forward(torch.cat([encoding, obs_act_encoding], dim=-1))

    @property
    def history_encoder(self) -> HistoryEncoder:
        """Action dimension."""
        return self._history_encoder
