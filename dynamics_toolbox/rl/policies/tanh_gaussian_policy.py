"""
Standard tanh Gaussian policy.

Author: Ian Char
Date: April 5, 2023
"""
from typing import Callable, Dict, Optional, Tuple, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

from dynamics_toolbox.rl.policies.abstract_policy import Policy
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
        """
        super().__init__(
            input_dim=obs_dim,
            output_dim=act_dim,
            hidden_sizes=hidden_sizes,
            hidden_activation=hidden_activation,
        )
        self.act_dim = act_dim
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
        normal_sample = dm.randn(size=mean.shape)
        actions = torch.tanh(normal_sample * std + mean)
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

    def get_action(self, obs_np: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Get action.

        Args:
            obs_np: numpy array of shape (obs_dim,)

        Returns:
            * Sampled actions.
            * Information dictionary.
        """
        actions, logprobs = self.get_actions(obs_np[None])
        return actions[0, :], {'logpi': float(logprobs)}

    def get_actions(self, obs_np: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get multiple actions.

        Args:
            obs_np: numpy array of shape (batch_size, obs_dim)

        Returns:
            * Sampled actions w shape (batch_size, act_dim)
            * Logprobabilities for the actions (batch_size,)
        """
        # Do forward pass.
        with torch.no_grad():
            actions, logprobs, mean, std = self.forward(dm.torch_ify(obs_np))
        # Sample actions.
        if self.deterministic:
            actions = torch.tanh(mean)
            logprobs = dm.ones(len(actions))
        return dm.get_numpy(actions), dm.get_numpy(logprobs.squeeze(-1))

    def train(self, mode: bool = True):
        super().train(mode)

    @property
    def act_dim(self) -> int:
        """Action dimension."""
        return self.act_dim
