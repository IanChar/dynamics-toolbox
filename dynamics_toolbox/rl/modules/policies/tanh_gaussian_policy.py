"""
Standard tanh Gaussian policy.

Author: Ian Char
Date: April 5, 2023
"""
from typing import Callable, Optional, Tuple, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

from dynamics_toolbox.rl.modules.history_encoders.abstract_history_encoder import (
    HistoryEncoder,
)
from dynamics_toolbox.rl.modules.policies.abstract_policy import Policy
from dynamics_toolbox.utils.pytorch.modules.fc_network import FCNetwork
from dynamics_toolbox.utils.pytorch.device_utils import MANAGER as dm


class TanhGaussianPolicy(FCNetwork, Policy):
    """Policy outputting tanh gaussian distribution."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: Sequence[int],
        hardcoded_std: Optional[float] = None,
        last_layer_uinit: float = 1e-3,
        hidden_activation: Callable[torch.Tensor, torch.Tensor] = F.relu,
        min_log_std: float = -20.0,
        max_log_std: float = 2.0,
        input_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
        **kwargs
    ):
        """Constructor.

        Args:
            obs_dim: Observation dimension size.
            act_dim: Action dimension size.
            hidden_sizes: Number of hidden units for each hidden layer.
            hardcoded_std: Hardcoded standard deviation. If none then the standard
                deviation is learned.
            last_layer_uinit: The bounds of the uniform distribution for the last
                layer. Initializing the layer to be 100x smaller has been shown
                to improve performance.
            hidden_activation: Activation function.
            min_log_std: The minimum log standard deviation that can be outputted.
            max_log_std: The maximum log standard deviation that can be outputted.
            input_dim: Input dimension if it is not the obs.
            output_dim: Output dimension if it is not the act.
        """
        super().__init__(
            input_dim=obs_dim if input_dim is None else input_dim,
            output_dim=act_dim if output_dim is None else output_dim,
            hidden_sizes=hidden_sizes,
            hidden_activation=hidden_activation,
        )
        self._obs_dim = obs_dim
        self._act_dim = act_dim
        self._min_log_std = min_log_std
        self._max_log_std = max_log_std
        self.deterministic = False
        # Initialize last layer to correct distribution.
        self.get_layer(self.n_layers - 1).weight.data.uniform_(
            -last_layer_uinit,
            last_layer_uinit,
        )
        self.get_layer(self.n_layers - 1).bias.data.uniform_(
            -last_layer_uinit,
            last_layer_uinit,
        )
        # Optionally create log standard deviation layer and initialize.
        self.std = hardcoded_std
        if self.std is None:
            penult = hidden_sizes[-1] if len(hidden_sizes) > 0 else obs_dim
            self._log_std_layer = nn.Linear(penult, act_dim)
            self._log_std_layer.weight.data.uniform_(
                -last_layer_uinit,
                last_layer_uinit,
            )
            self._log_std_layer.bias.data.uniform_(
                -last_layer_uinit,
                last_layer_uinit,
            )

    def forward(self, obs: Tensor) -> Tuple[Tensor]:
        """Forward pass of the network.

        Args:
            obs: The observations as tensor of shape (batch_size, obs_dim).

        Returns: Several tensors.
            * actions: Sampled actions as tensor of shape (batch_size, act_dim)
            * logprobs: Log probabilities of the actions w shape (batch_size, 1)
            * means: Mean of the distribution w shape (batch_size, act_dim)
            * stds: Standard dev of the distribution w shape (batch_size, act_dim)
        """
        # Get output of the networks.
        curr = obs
        for layer_num in range(self.n_layers - 1):
            curr = self.get_layer(layer_num)(curr)
            curr = self._hidden_activation(curr)
        mean = self.get_layer(self.n_layers - 1)(curr)
        if self.std is None:
            log_std = torch.clamp(
                self._log_std_layer(curr),
                self._min_log_std,
                self._max_log_std,
            )
            std = torch.exp(log_std)
        else:
            std = self.std
            log_std = torch.log(std)
        # Create samples.
        normal_sample = dm.randn(size=mean.shape) * std + mean
        actions = torch.tanh(normal_sample)
        logprobs = torch.sum(
            -0.5 * ((normal_sample - mean) / std).pow(2)
            - torch.log(std)
            - 0.5 * dm.from_numpy(np.log([2.0 * np.pi])),
            dim=-1,
            keepdim=True,
        )
        logprobs -= 2.0 * (
            dm.from_numpy(np.log([2.]))
            - normal_sample
            - torch.nn.functional.softplus(-2.0 * normal_sample)
        ).sum(dim=-1, keepdim=True)
        return actions, logprobs, mean, std

    def get_actions(self, obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get multiple actions.

        If batch_size is 1 for any of these we squeeze it.

        Args:
            obs: numpy array of shape (batch_size, obs_dim)

        Returns:
            * Sampled actions w shape (batch_size, act_dim)
            * Logprobabilities for the actions (batch_size,)
        """
        if len(obs.shape) == 1:
            obs = obs[np.newaxis]
        # Do forward pass.
        with torch.no_grad():
            actions, logprobs, mean, std = self.forward(dm.torch_ify(obs))
        # Sample actions.
        if self.deterministic:
            actions = torch.tanh(mean)
            logprobs = dm.ones(len(actions))
        return dm.get_numpy(actions.squeeze()), dm.get_numpy(logprobs.squeeze())

    def train(self, mode: bool = True):
        super().train(mode)

    @property
    def act_dim(self) -> int:
        """Action dimension."""
        return self._act_dim


class SequentialTanhGaussianPolicy(TanhGaussianPolicy):

    def __init__(
        self,
        history_encoder: HistoryEncoder,
        obs_encode_dim: int,
        **kwargs
    ):
        super().__init__(
            input_dim=history_encoder.out_dim + obs_encode_dim,
            **kwargs
        )
        self._history_encoder = history_encoder
        self._obs_encoder = nn.Linear(self._obs_dim, obs_encode_dim)
        self.reset()

    def reset(self):
        """Reset by setting the histories to None."""
        self._encode_history = None
        self._obs_history = None
        self._act_history = None
        self._rew_history = None

    def forward(self, obs_seq, act_seq, rew_seq, history=None):
        """
        Forward pass. Where B = batch_size, L = seq length..
            * obs_seq: (B, L, obs_dim)
            * act_seq: (B, L, act_dim)
            * rew_seq: (B, L, 1)

        Outputs:
            * actions (B, L, act_dim)
            * logprobs (B, L, 1)
            * means: (B, L, act_dim)
            * stds: (B, L, act_dim)
            * history: Depends on the sequence encoder.
        """
        encoding, history = self._history_encoder(obs_seq, act_seq, rew_seq, history)
        obs_encoding = self.hidden_activation(self._obs_encoder(
            obs_seq[:, -encoding.shape[1]:]))
        if encoding.shape[1] == 1:
            obs_encoding = obs_encoding[:, [-1]]
        actions, logprobs, means, stds = super().forward(torch.cat([
            encoding, obs_encoding], dim=-1))
        return actions, logprobs, means, stds, history

    def get_actions(self, obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get multiple actions.

        If batch_size is 1 for any of these we squeeze it.

        Args:
            obs: numpy array of shape (batch_size, obs_dim)

        Returns:
            * Sampled actions w shape (batch_size, act_dim)
            * Logprobabilities for the actions (batch_size,)
        """
        if len(obs.shape) == 1:
            obs = obs[np.newaxis]
        obs = dm.torch_ify(obs).unsqueeze(1)
        if self._obs_history is None:
            self._obs_history = obs
            self._act_history = dm.zeros(len(obs), 1, self.act_dim)
            self._rew_history = dm.zeros(len(obs), 1, 1)
        else:
            self._obs_history = torch.cat([self._obs_history, obs], dim=1)
        with torch.no_grad():
            actions, logprobs, mean, std, history = self.forward(
                self._obs_history,
                self._act_history,
                self._rew_history,
                history=self._encode_history,
            )
        actions, logprobs, mean, std = [seq[:, -1] for seq in
                                        (actions, logprobs, mean, std)]
        self._encode_history = history
        # Sample actions.
        if self.deterministic:
            actions = torch.tanh(mean)
            logprobs = dm.ones(len(actions))
        self._act_history = torch.cat([self._act_history, actions.unsqueeze(1)],
                                      dim=1)
        return dm.get_numpy(actions.squeeze()), dm.get_numpy(logprobs.squeeze())

    def get_reward_feedback(self, rewards: Union[float, np.ndarray]):
        """Get feedback from the environment about the last reward.

        Args:
            rewards: The rewards as a float or list of rewards if doing multiple
                     rollouts.
        """
        pass
        if isinstance(rewards, float):
            rewards = np.array([rewards])
        rewards = dm.from_numpy(rewards).view(-1, 1, 1)
        if self._rew_history is None:
            self._rew_history = rewards
        else:
            self._rew_history = torch.cat([self._rew_history, rewards], dim=1)
