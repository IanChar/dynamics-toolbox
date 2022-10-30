"""
Recursive model that predicts a gaussian distribtion.

Author: Ian Char
Date: 10/27/2022
"""
from typing import Dict, Callable, Tuple, Any, Sequence, Optional

import hydra.utils
import torch
from omegaconf import DictConfig
from torchmetrics import ExplainedVariance

from dynamics_toolbox.constants import losses, sampling_modes
from dynamics_toolbox.models.pl_models.sequential_models.abstract_sequential_model \
        import AbstractSequentialModel
from dynamics_toolbox.utils.pytorch.losses import get_regression_loss


class RPNN(AbstractSequentialModel):
    """RPNN network."""

    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            encode_dim: int,
            rnn_num_layers: int,
            rnn_hidden_size: int,
            encoder_cfg: DictConfig,
            pnn_decoder_cfg: DictConfig,
            rnn_type: str = 'gru',
            warm_up_period: int = 0,
            learning_rate: float = 1e-3,
            logvar_lower_bound: Optional[float] = None,
            logvar_upper_bound: Optional[float] = None,
            logvar_bound_loss_coef: float = 1e-3,
            sample_mode: str = sampling_modes.SAMPLE_FROM_DIST,
            weight_decay: Optional[float] = 0.0,
            use_layer_norm: bool = True,
            **kwargs,
    ):
        """Constructor.

        Args:
            input_dim: The input dimension.
            output_dim: The output dimension.
            encode_dim: The dimension of the encoder output.
            rnn_num_layers: Number of layers in the memory unit.
            rnn_hidden_size: The number hidden units in the memory unit.
            encoder_cfg: The configuration for the encoder network.
            pnn_decoder_cfg: The configuration for the decoder network. Should
                be a PNN.
            rnn_type: Name of the rnn type. Can accept GRU or LSTM.
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
            use_layer_norm: Whether to use layer norm.
        """
        super().__init__(input_dim, output_dim, **kwargs)
        self.save_hyperparameters()
        self._encoder = hydra.utils.instantiate(
            encoder_cfg,
            input_dim=input_dim,
            output_dim=encode_dim,
            _recursive_=False,
        )
        self._decoder = hydra.utils.instantiate(
            pnn_decoder_cfg,
            input_dim=encode_dim + rnn_hidden_size,
            output_dim=output_dim,
            _recursive_=False,
        )
        if rnn_type.lower() == 'gru':
            rnn_class = torch.nn.GRU
        elif rnn_type.lower() == 'lstm':
            rnn_class = torch.nn.LSTM
        else:
            raise ValueError(f'Cannot recognize RNN type {rnn_type}')
        self._memory_unit = rnn_class(encode_dim, rnn_hidden_size,
                                      num_layers=rnn_num_layers,
                                      batch_first=True,
                                      device=self.device)
        if use_layer_norm:
            self._layer_norm = torch.nn.LayerNorm(encode_dim)
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._encode_dim = encode_dim
        self._hidden_size = rnn_hidden_size
        self._num_layers = rnn_num_layers
        self._warm_up_period = warm_up_period
        self._learning_rate = learning_rate
        self._weight_decay = weight_decay
        self._sample_mode = sample_mode
        self._record_history = True
        self._hidden_state = None
        self._use_layer_norm = use_layer_norm
        # Set up variance pinning.
        self._var_pinning = (logvar_lower_bound is not None
                             and logvar_upper_bound is not None)
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
        self._logvar_bound_loss_coef = logvar_bound_loss_coef
        # TODO: In the future we may want to pass this in as an argument.
        self._metrics = {
            'EV': ExplainedVariance(),
            'IndvEV': ExplainedVariance('raw_values'),
        }

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
        encoded = self._encoder(batch[0])
        if self._use_layer_norm:
            encoded = self._layer_norm(encoded)
        mem_out = self._memory_unit(encoded)[0]
        mean, logvar = self._decoder(torch.cat([encoded, mem_out], dim=-1))
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
        if self._hidden_state is None:
            self._hidden_state = torch.zeros(self._num_layers, net_in.shape[0],
                                             self._hidden_size, device=self.device)
        elif self._hidden_state.shape[1] != net_in.shape[0]:
            raise ValueError('Number of inputs does not match previously given number.'
                             f' Expected {self._hidden_state.shape[1]} but received'
                             f' {net_in.shape[0]}.')
        with torch.no_grad():
            encoded = self._encoder(net_in.unsqueeze(0))
            if self._use_layer_norm:
                encoded = self._layer_norm(encoded)
            mem_out, hidden_out = self._memory_unit(encoded, self._hidden_state)
            if self._record_history:
                self._hidden_state = hidden_out
            mean_predictions, logvar_predictions =\
                self._decoder(torch.cat([encoded, mem_out], dim=-1)).squeeze(0)
        std_predictions = (0.5 * logvar_predictions).exp()
        if self._sample_mode == sampling_modes.SAMPLE_FROM_DIST:
            predictions = (torch.randn_like(mean_predictions) * std_predictions
                           + mean_predictions)
        else:
            predictions = mean_predictions
        info = {'predictions': predictions,
                'mean_predictions': mean_predictions,
                'std_predictions': std_predictions}
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
        return self._sample_mode

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
        self._hidden_state = None

    def reset(self) -> None:
        """Reset the dynamics model."""
        self.clear_history()
