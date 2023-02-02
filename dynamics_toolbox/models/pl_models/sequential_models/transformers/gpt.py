"""
GPT style model.

Author: Ian Char
Date: January 30, 2023
"""
import math
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import hydra
from omegaconf import DictConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import ExplainedVariance

from dynamics_toolbox.constants import sampling_modes
from dynamics_toolbox.models.pl_models.sequential_models.abstract_sequential_model \
        import AbstractSequentialModel
from dynamics_toolbox.utils.pytorch.modules.attention import GPTBlock


class GPT(AbstractSequentialModel):
    """GPT architecture.

    Code inspired by Karpathy's nanoGPT.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_blocks: int,
        embed_dim_per_head: int,
        n_heads: int,
        block_size: int,
        posn_embedding: Optional[DictConfig],
        dropout: float = 0.0,
        bias: bool = False,
        warm_up_period: int = 0,
        learning_rate: float = 3e-4,
        logvar_lower_bound: Optional[float] = None,
        logvar_upper_bound: Optional[float] = None,
        logvar_bound_loss_coef: float = 1e-3,
        sample_mode: str = sampling_modes.SAMPLE_FROM_DIST,
        weight_decay: Optional[float] = 0.0,
        **kwargs,
    ):
        """Constructor.

        Args:
            num_blocks: The number of blocks.
            embed_dim_per_head: The dimension of the embedding per head.
            n_heads: The number of heads.
            block_size: Size of the block, i.e. the sequence length.
            posn_embedding: The configuration for the positional embedding.
            dropout: The amount of dropout.
            bias: Whether to have bias in the linear layer.
            warm_up_period: The amount of data to take in before predictions begin to
                be made.
            learning_rate: The learning rate for the network.
            logvar_lower_bound: Lower bound on the log variance.
                If none there is no bound.
            logvar_upper_bound: Lower bound on the log variance.
                If none there is no bound.
            logvar_bound_loss_coef: Coefficient on bound loss to add to loss.
            sample_mode: The method to use for sampling.
            weight_decay: The weight decay for the optimizer.
        """
        super().__init__(input_dim, output_dim, **kwargs)
        self._input_dim = input_dim
        self._output_dim = output_dim
        self.embed_dim = embed_dim_per_head * n_heads
        self.block_size = block_size
        self._learning_rate = learning_rate
        self._weight_decay = weight_decay
        self._sample_mode = sample_mode
        self._record_history = True
        self._history = None
        self._time_step = 0
        self._warm_up_period = warm_up_period
        self._var_pinning = (logvar_lower_bound is not None
                             and logvar_upper_bound is not None)
        self._logvar_bound_loss_coef = logvar_bound_loss_coef
        self._metrics = {
            'EV': ExplainedVariance(),
            'IndvEV': ExplainedVariance('raw_values'),
        }
        # Initialize the modules.
        if posn_embedding is None:
            self.posn_embedder = None
        else:
            self.posn_embedder = hydra.utils.instantiate(posn_embedding)
        self.encoder = nn.Linear(input_dim, self.embed_dim, bias=False)
        self.blocks = nn.ModuleList([GPTBlock(
            embed_dim_per_head=embed_dim_per_head,
            n_heads=n_heads,
            block_size=block_size,
            dropout=dropout,
            bias=bias,
        ) for _ in range(num_blocks)])
        self.mean_decoder = nn.Linear(self.embed_dim, output_dim, bias=False)
        self.std_decoder = nn.Linear(self.embed_dim, output_dim, bias=False)
        if self._var_pinning:
            self._min_logvar = torch.nn.Parameter(
                torch.Tensor([logvar_lower_bound])
                * torch.ones(1, output_dim, dtype=torch.float32, requires_grad=True))
            self._max_logvar = torch.nn.Parameter(
                torch.Tensor([logvar_upper_bound])
                * torch.ones(1, output_dim, dtype=torch.float32, requires_grad=True))
        else:
            self._min_logvar = None
            self._max_logvar = None
        # Initialize the weights.
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * num_blocks))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward function for network

        Args:
            x: The input to the network with shape (batch_size, seq_length, dim)

        Returns:
            The mean and log variance with the same shapes as input.
        """
        encoded = self.encoder(x)
        if self.posn_embedder is not None:
            encoded = self.posn_embedder(encoded)
        for block in self.blocks:
            encoded = block(encoded)
        mean = self.mean_decoder(encoded)
        logvar = self.std_decoder(encoded)
        if self._var_pinning:
            logvar = self._max_logvar - F.softplus(self._max_logvar - logvar)
            logvar = self._min_logvar + F.softplus(logvar - self._min_logvar)
        return mean, logvar

    def get_net_out(self, batch: Sequence[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Get the output of the network and organize into dictionary.

        Args:
            batch: The batch passed into the network. This is expected to be a tuple
                * x: (Batch_size, Sequence Length, dim)
                * y: (Batch_size, Sequence Length, dim)
                * mask: (Batch_size, Sequence Length, 1)

        Returns:
            Dictionary of name to tensor.
        """
        mean, logvar = self.get_net_out(batch[0])
        return {'mean': mean, 'logvar': logvar}

    def loss(self, net_out: Dict[str, torch.Tensor], batch: Sequence[torch.Tensor]) -> \
            Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute the loss function.

        Args:
            net_out: The output of the network.
            batch: The batch passed into the network. This is expected to be a tuple
                * x: (Batch_size, Sequence Length, dim)
                * y: (Batch_size, Sequence Length, dim)
                * mask: (Batch_size, Sequence Length, 1)

        Returns:
            The loss and a dictionary of other statistics.
        """
        mean = net_out['mean']
        logvar = net_out['logvar']
        y, mask = batch[1:]
        mask[:, :self._warm_up_period, :] = 0
        sq_diffs = (mean * mask - y * mask).pow(2)
        mse = torch.mean(sq_diffs)
        loss = torch.mean(torch.exp(-logvar) * sq_diffs + logvar * mask)
        stats = dict(
            nll=loss.item(),
            mse=mse.item(),
        )
        stats['logvar/mean'] = (logvar * mask).mean().item()
        if self._var_pinning:
            bound_loss = self._logvar_bound_loss_coef * \
                         torch.abs(self._max_logvar - self._min_logvar).mean()
            stats['bound_loss'] = bound_loss.item()
            stats['logvar_lower_bound/mean'] = self._min_logvar.mean().item()
            stats['logvar_upper_bound/mean'] = self._max_logvar.mean().item()
            stats['logvar_bound_difference'] = (
                        self._max_logvar - self._min_logvar).mean().item()
            loss += bound_loss
        stats['loss'] = loss.item()
        return loss, stats

    def single_sample_output_from_torch(self, net_in: torch.Tensor) -> Tuple[
            torch.Tensor, Dict[str, Any]]:
        """Get the output for a single sample in the model.

        Args:
            net_in: The input for the network with expected shape (batch size, dim)

        Returns:
            The predictions for a single function sample.
        """
        if self._history is None:
            self._history = torch.zeros(net_in.shape[0], self.block_size,
                                        net_in.shape[1])
        elif self._history.shape[0] != net_in.shape[0]:
            raise ValueError('Number of inputs does not match previously given number.'
                             f' Expected {self._history.shape[0]} but received'
                             f' {net_in.shape[0]}.')
        if self._time_step >= self.block_size - 1:
            self._history = torch.cat([
                self._history[:, 1:],
                net_in.unsqueeze(1),
            ], dim=1)
        else:
            self._history[:, self._time_step] = net_in
        with torch.no_grad():
            out_stats = self.get_net_out([self._history])
        pred_idx = min(self._time_step, out_stats['mean'].shape[1] - 1)
        mean_predictions = out_stats['mean'][:, pred_idx]
        logvar_predictions = out_stats['logvar'][:, pred_idx]
        std_predictions = (0.5 * logvar_predictions).exp()
        if self._sample_mode == sampling_modes.SAMPLE_FROM_DIST:
            predictions = (torch.randn_like(mean_predictions) * std_predictions
                           + mean_predictions)
        else:
            predictions = mean_predictions
        info = {
            'predictions': predictions,
            'mean_predictions': mean_predictions,
            'std_predictions': std_predictions,
            'all_mean_predictions': out_stats['mean'],
            'all_logvar_predictions': out_stats['logvar']
        }
        self._time_step += 1
        return predictions, info

    def multi_sample_output_from_torch(self, net_in: torch.Tensor) -> Tuple[
            torch.Tensor, Dict[str, Any]]:
        """Get the output where each input is assumed to be from a different sample.

        Args:
            net_in: The input for the network.

        Returns:
            The deltas for next states and dictionary of info.
        """
        return self.single_sample_output_from_torch(net_in)

    def _init_weights(self, module: torch.nn.Module):
        """Intialize the weights of a module."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    @property
    def metrics(self) -> Dict[str, Callable[[torch.Tensor], torch.Tensor]]:
        return self._metrics

    @property
    def learning_rate(self) -> float:
        return self._learning_rate

    @property
    def weight_decay(self) -> float:
        return self._weight_decay

    @property
    def sample_mode(self) -> str:
        """The sample mode is the method that in which we get next state."""
        return self._sample_mode

    @sample_mode.setter
    def sample_mode(self, mode: str) -> None:
        """Set the sample mode to the appropriate mode."""
        if self._sample_mode not in [sampling_modes.SAMPLE_FROM_DIST,
                                     sampling_modes.RETURN_MEAN]:
            raise ValueError(
                f'PNN sample mode must either be {sampling_modes.SAMPLE_FROM_DIST} '
                f'or {sampling_modes.RETURN_MEAN}, but received {mode}.')
        self._sample_mode = mode

    @property
    def input_dim(self) -> int:
        return self._input_dim

    @property
    def output_dim(self) -> int:
        return self._output_dim

    @property
    def record_history(self) -> bool:
        """Whether to keep track of the quantities being fed into the neural net."""
        return self._record_history

    @record_history.setter
    def record_history(self, mode: bool) -> None:
        """Set whether to keep track of quantities being fed into the neural net."""
        self._record_history = mode

    @property
    def warm_up_period(self) -> int:
        """Amount of data to take in before starting to predict"""
        return self._warm_up_period

    def clear_history(self) -> None:
        """Clear the history."""
        self._history = None
        self._time_step = 0

    def reset(self) -> None:
        """Reset the dynamics model."""
        self.clear_history()
