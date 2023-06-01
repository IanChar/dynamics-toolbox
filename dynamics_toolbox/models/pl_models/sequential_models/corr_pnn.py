"""
PNN that is temporally corelated.
"""
from typing import Dict, Callable, Tuple, Any, Sequence, Optional

import torch
import torch.nn.functional as F

from dynamics_toolbox.constants import sampling_modes
from dynamics_toolbox.models.pl_models.sequential_models.abstract_sequential_model \
        import AbstractSequentialModel
from dynamics_toolbox.utils.pytorch.activations import identity
from dynamics_toolbox.utils.pytorch.metrics import SequentialExplainedVariance
from dynamics_toolbox.utils.pytorch.modules.fc_network import FCNetwork


class CorrPNN(AbstractSequentialModel):
    """RPNN network."""

    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            encode_dim: int,
            rnn_hidden_size: int,
            decoder_num_hidden_layers: int,
            decoder_hidden_size: int,
            encoder_num_hidden_layers: int = 0,
            encoder_hidden_size: int = 0,
            rnn_num_layers: int = 1,
            rnn_type: str = 'gru',
            learning_rate: float = 1e-3,
            logvar_lower_bound: Optional[float] = None,
            logvar_upper_bound: Optional[float] = None,
            logvar_bound_loss_coef: float = 1e-3,
            corr_lower_bound: Optional[float] = -0.05,
            corr_upper_bound: Optional[float] = 0.05,
            corr_bound_loss_coef: float = 1e-3,
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
            rnn_hidden_size: The number hidden units in the memory unit.
            decoder_num_hidden_layers: The number of hidden layers for the decoder.
            decoder_hidden_size: Number of hidden units per layer in the decoder.
            encoder_num_hidden_layers: The number of hidden layers for the encoder.
            encoder_hidden_size: Number of hidden units per layer in the encoder.
            rnn_num_layers: Number of layers in the memory unit.
            rnn_type: Name of the rnn type. Can accept GRU or LSTM.
            learning_rate: The learning rate for the network.
            logvar_lower_bound: Lower bound on the log variance.
                If none there is no bound.
            logvar_upper_bound: Lower bound on the log variance.
                If none there is no bound.
            logvar_bound_loss_coef: Coefficient on bound loss to add to loss.
            corr_lower_bound: Lower bound on the correlation.
                If none there is no bound.
            corr_upper_bound: Lower bound on the correlation.
                If none there is no bound.
            corr_bound_loss_coef: Coefficient on bound loss to add to loss.
            sample_mode: The method to use for sampling.
            weight_decay: The weight decay for the optimizer.
            use_layer_norm: Whether to use layer norm.
        """
        super().__init__(input_dim, output_dim, **kwargs)
        if encoder_num_hidden_layers > 0:
            encoder_hidden_sizes = [encoder_hidden_size
                                    for _ in range(encoder_num_hidden_layers)]
        else:
            encoder_hidden_sizes = None
        decoder_hidden_sizes = [decoder_hidden_size
                                for _ in range(decoder_num_hidden_layers)]
        self._encoder = FCNetwork(
            input_dim=input_dim,
            output_dim=encode_dim,
            hidden_sizes=encoder_hidden_sizes,
        )
        self._decoder = FCNetwork(
            input_dim=encode_dim + rnn_hidden_size,
            output_dim=output_dim,
            hidden_sizes=decoder_hidden_sizes,
            num_heads=3,
            out_activation=[identity, identity, torch.tanh],
        )
        self.rnn_type = rnn_type.lower()
        if rnn_type.lower() == 'gru':
            rnn_class = torch.nn.GRU
        elif rnn_type.lower() == 'lstm':
            rnn_class = torch.nn.LSTM
        else:
            raise ValueError(f'Cannot recognize RNN type {rnn_type}')
        self._memory_unit = rnn_class(encode_dim, rnn_hidden_size,
                                      num_layers=rnn_num_layers,
                                      batch_first=True)
        self._memory_unit = self._memory_unit.to(self.device)
        if use_layer_norm:
            self._layer_norm = torch.nn.LayerNorm(encode_dim)
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._encode_dim = encode_dim
        self._hidden_size = rnn_hidden_size
        self._num_layers = rnn_num_layers
        self._learning_rate = learning_rate
        self._weight_decay = weight_decay
        self._sample_mode = sample_mode
        self._record_history = True
        self._hidden_state = None
        self._last_pred_info = None
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
        # Set up correlation pinning.
        self._corr_pinning = (corr_lower_bound is not None
                              and corr_upper_bound is not None)
        if self._corr_pinning:
            self._min_corr = torch.nn.Parameter(
                torch.Tensor([corr_lower_bound])
                * torch.ones(1, output_dim, dtype=torch.float32, requires_grad=True))
            self._max_corr = torch.nn.Parameter(
                torch.Tensor([corr_upper_bound])
                * torch.ones(1, output_dim, dtype=torch.float32, requires_grad=True))
        else:
            self._min_corr = None
            self._max_corr = None
        self._corr_bound_loss_coef = corr_bound_loss_coef
        # TODO: In the future we may want to pass this in as an argument.
        self._metrics = {
            'EV': SequentialExplainedVariance(),
            'IndvEV': SequentialExplainedVariance('raw_values'),
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
                * mean (batch_size, Sequence Length, dim)
                * logvar (batch_size, Sequence Length, dim)
                * corr (batch_size, Sequence Length, dim)
        """
        encoded = self._encoder(batch[0])
        if self._use_layer_norm:
            encoded = self._layer_norm(encoded)
        mem_out = self._memory_unit(encoded)[0]
        mean, logvar, corr = self._decoder(torch.cat([encoded, mem_out], dim=-1))
        if self._var_pinning:
            logvar = self._max_logvar - F.softplus(self._max_logvar - logvar)
            logvar = self._min_logvar + F.softplus(logvar - self._min_logvar)
        if self._corr_pinning:
            corr = self._max_corr - F.softplus(self._max_corr - corr, beta=100)
            corr = self._min_corr + F.softplus(corr - self._min_corr, beta=100)
        return {'mean': mean, 'logvar': logvar, 'corr': corr}

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
        corr = net_out['corr']
        y, mask = batch[1:]
        num_valid = mask[:, 1:].sum().item()
        num_valid_full = mask.sum().item()
        diffs = y - mean
        sq_diffs = diffs.pow(2)
        norm_sq_diffs = sq_diffs * torch.exp(-logvar)
        corr_term = 1 - corr[:, 1:].pow(2)
        loss = ((
            (2 * corr_term).pow(-1) * (
                norm_sq_diffs[:, :-1]
                + norm_sq_diffs[:, 1:]
                - (2 * corr[:, 1:] * diffs[:, :-1] * diffs[:, 1:]
                   * torch.exp(-0.5 * logvar[:, 1:] * logvar[:, :-1]))
            )
            + logvar[:, :-1]
            + logvar[:, 1:]
            + 0.5 * corr_term.log()
        ) * mask[:, 1:]).mean(dim=-1).sum() / num_valid
        stats = {}
        stats['nll'] = loss.item()
        stats['mse'] = (sq_diffs * mask).mean(dim=-1).sum().item() / num_valid_full
        stats['logvar/mean'] = ((logvar * mask).mean(dim=-1).sum()
                                / num_valid_full).item()
        stats['correlation/mean'] = ((corr[:, 1:] * mask[:, 1:]).mean(dim=-1).sum()
                                     / num_valid).item()
        stats['correlation/min'] = (corr[:, 1:] * mask[:, 1:]).min().item()
        stats['correlation/max'] = (corr[:, 1:] * mask[:, 1:]).max().item()
        if self._var_pinning:
            bound_loss = self._logvar_bound_loss_coef * \
                         torch.abs(self._max_logvar - self._min_logvar).mean()
            stats['logvar_bound_loss'] = bound_loss.item()
            stats['logvar_lower_bound/mean'] = self._min_logvar.mean().item()
            stats['logvar_upper_bound/mean'] = self._max_logvar.mean().item()
            stats['logvar_bound_difference'] = (
                        self._max_logvar - self._min_logvar).mean().item()
            loss += bound_loss
        if self._corr_pinning:
            bound_loss = self._corr_bound_loss_coef * \
                         torch.abs(self._max_corr - self._min_corr).mean()
            stats['corr_bound_loss'] = bound_loss.item()
            stats['corr_lower_bound/mean'] = self._min_corr.mean().item()
            stats['corr_upper_bound/mean'] = self._max_corr.mean().item()
            stats['corr_bound_difference'] = (
                        self._max_corr - self._min_corr).mean().item()
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
            if self.rnn_type == 'gru':
                self._hidden_state = torch.zeros(self._num_layers, net_in.shape[0],
                                                 self._hidden_size, device=self.device)
            else:
                self._hidden_state = tuple(
                    torch.zeros(self._num_layers, net_in.shape[0],
                                self._hidden_size, device=self.device)
                    for _ in range(2))
        else:
            tocompare = (self._hidden_state if self.rnn_type == 'gru'
                         else self._hidden_state[0])
            if tocompare.shape[1] != net_in.shape[0]:
                raise ValueError('Number of inputs does not match previously given '
                                 f'number. Expected {tocompare.shape[1]} but received'
                                 f' {net_in.shape[0]}.')
        with torch.no_grad():
            encoded = self._encoder(net_in).unsqueeze(1)
            if self._use_layer_norm:
                encoded = self._layer_norm(encoded)
            mem_out, hidden_out = self._memory_unit(encoded, self._hidden_state)
            if self._record_history:
                self._hidden_state = hidden_out
            mean_predictions, logvar_predictions, corr_predictions =\
                (output.squeeze(1) for output in
                 self._decoder(torch.cat([encoded, mem_out], dim=-1)))
        std_predictions = (0.5 * logvar_predictions).exp()
        if self._sample_mode == sampling_modes.SAMPLE_FROM_DIST:
            if self._last_pred_info is not None:
                mean_predictions += (
                    corr_predictions * std_predictions
                    / self._last_pred_info['std']
                    * (self._last_pred_info['samples'] - self._last_pred_info['mean']))
                std_predictions *= (1 - corr_predictions.pow(2)).sqrt()
            predictions = (torch.randn_like(mean_predictions) * std_predictions
                           + mean_predictions)
            if self._record_history:
                self._last_pred_info = {
                    'mean': mean_predictions,
                    'std': std_predictions,
                    'samples': predictions,
                }
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
        self._hidden_state = None
        self._last_pred_info = None

    def reset(self) -> None:
        """Reset the dynamics model."""
        self.clear_history()
