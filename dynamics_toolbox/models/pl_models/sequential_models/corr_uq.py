"""
Model that wraps separate model and learns additional UQ over it. This UQ may help w

1. Calibration.
2. Correlated Errors.

Author: Ian Char
Date: April 27, 2023
"""
from typing import Any, Dict, Callable, Tuple, Sequence, Optional

import hydra.utils
import torch
from omegaconf import DictConfig

from dynamics_toolbox.constants import sampling_modes
from dynamics_toolbox.models.pl_models.sequential_models.abstract_sequential_model \
        import AbstractSequentialModel
from dynamics_toolbox.utils.pytorch.metrics import SequentialExplainedVariance
from dynamics_toolbox.utils.pytorch.modules.fc_network import FCNetwork


class CorrUQ(AbstractSequentialModel):

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        base_model: DictConfig,
        base_model_is_sequential: bool,
        uq_encode_dim: int,
        uq_rnn_hidden_size: int,
        uq_decoder_num_hidden_layers: int,
        uq_decoder_hidden_size: int,
        uq_encoder_num_hidden_layers: int = 0,
        uq_encoder_hidden_size: int = 0,
        uq_rnn_num_layers: int = 1,
        uq_rnn_type: str = 'gru',
        base_learning_rate: float = 1e-3,
        uq_learning_rate: float = 5e-4,
        corr_max_magnitude: float = 0.9,
        sample_mode: str = sampling_modes.SAMPLE_FROM_DIST,
        base_weight_decay: Optional[float] = 0.0,
        uq_weight_decay: Optional[float] = 0.0,
        **kwargs
    ):
        """Constructor."""
        super().__init__(input_dim, output_dim, **kwargs)
        self._base_model = hydra.utils.instantiate(
            base_model,
            input_dim=input_dim,
            output_dim=output_dim,
            _recursive_=False,
        )
        if uq_encoder_num_hidden_layers > 0:
            encoder_hidden_sizes = [uq_encoder_hidden_size
                                    for _ in range(uq_encoder_num_hidden_layers)]
        else:
            encoder_hidden_sizes = None
        decoder_hidden_sizes = [uq_decoder_hidden_size
                                for _ in range(uq_decoder_num_hidden_layers)]
        self._encoder = FCNetwork(
            input_dim=input_dim,
            output_dim=uq_encode_dim,
            hidden_sizes=encoder_hidden_sizes,
        )
        self._decoder = FCNetwork(
            input_dim=uq_encode_dim + uq_rnn_hidden_size,
            output_dim=output_dim,
            hidden_sizes=decoder_hidden_sizes,
            out_activation=torch.tanh,
        )
        self.uq_rnn_type = uq_rnn_type.lower()
        if uq_rnn_type.lower() == 'gru':
            rnn_class = torch.nn.GRU
        elif uq_rnn_type.lower() == 'lstm':
            rnn_class = torch.nn.LSTM
        else:
            raise ValueError(f'Cannot recognize RNN type {uq_rnn_type}')
        self._memory_unit = rnn_class(uq_encode_dim, uq_rnn_hidden_size,
                                      num_layers=uq_rnn_num_layers,
                                      batch_first=True)
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._base_is_seq = base_model_is_sequential
        self._encode_dim = uq_encode_dim
        self._hidden_size = uq_rnn_hidden_size
        self._num_layers = uq_rnn_num_layers
        self._base_learning_rate = base_learning_rate
        self._uq_learning_rate = uq_learning_rate
        self._base_weight_decay = base_weight_decay
        self._uq_weight_decay = uq_weight_decay
        self._sample_mode = sample_mode
        self._corr_max_magnitude = corr_max_magnitude
        self._record_history = True
        self._hidden_state = None
        self._last_pred_info = None
        self._metrics = {
            'EV': SequentialExplainedVariance(),
            'IndvEV': SequentialExplainedVariance('raw_values'),
        }

    def training_step(
            self,
            batch: Sequence[torch.Tensor],
            batch_idx: int,
            optimizer_idx: int,
    ) -> torch.Tensor:
        """Training step for pytorch lightning. Returns the loss."""
        batch = self.normalizer.normalize_batch(batch)
        if optimizer_idx == 0:  # Base model.
            if not self._base_is_seq:
                base_batch = [b.reshape(-1, b.shape[-1]) for b in batch]
            base_out = self._base_model.get_net_out(base_batch)
            base_loss, base_dict = self._base_model.loss(base_out, base_batch)
            self._log_stats(base_dict, prefix='train')
            return base_loss
        net_out = self.get_net_out(batch)
        loss, loss_dict = self.loss(net_out, batch)
        self._log_stats(loss_dict, prefix='train')
        return loss

    def validation_step(
        self,
        batch: Sequence[torch.Tensor],
        batch_idx: int,
    ) -> None:
        """Training step for pytorch lightning. Returns the loss."""
        batch = self.normalizer.normalize_batch(batch)
        # Log base model stats.
        if not self._base_is_seq:
            base_batch = [b.reshape(-1, b.shape[-1]) for b in batch]
        base_out = self._base_model.get_eval_net_out(base_batch)
        base_loss, base_dict = self._base_model.loss(base_out, base_batch)
        base_dict.update(self._base_model._get_test_and_validation_metrics(
            base_out, base_batch))
        self._log_stats(base_dict, prefix='val')
        # Log UQ stats.
        net_out = self.get_eval_net_out(batch)
        loss, loss_dict = self.loss(net_out, batch)
        loss_dict.update(self._get_test_and_validation_metrics(net_out, batch))
        self._log_stats(loss_dict, prefix='val')

    def test_step(
        self,
        batch: Sequence[torch.Tensor],
        batch_idx: int,
    ) -> None:
        """Training step for pytorch lightning. Returns the loss."""
        batch = self.normalizer.normalize_batch(batch)
        # Log base model stats.
        if not self._base_is_seq:
            base_batch = [b.reshape(-1, b.shape[-1]) for b in batch]
        base_out = self._base_model.get_eval_net_out(base_batch)
        base_loss, base_dict = self._base_model.loss(base_out, base_batch)
        base_dict.update(self._base_model._get_test_and_validation_metrics(
            base_out, base_batch))
        self._log_stats(base_dict, prefix='test')
        # Log UQ stats.
        net_out = self.get_eval_net_out(batch)
        loss, loss_dict = self.loss(net_out, batch)
        loss_dict.update(self._get_test_and_validation_metrics(net_out, batch))
        self._log_stats(loss_dict, prefix='test')

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
        B, L, D_y = batch[1].shape
        # Forward pass of base model.
        # TODO: This happens twice per step bc of pl. Is there a way of caching?
        with torch.no_grad():
            if self._base_is_seq:
                base_out = self._base_model.get_net_out(batch)
                assert 'mean' in base_out and 'std' in base_out
            else:
                base_out = self._base_model.get_net_out([b.reshape(-1, b.shape[-1])
                                                         for b in batch])
                assert 'mean' in base_out and 'std' in base_out
                base_out = {
                    'mean': base_out['mean'].reshape(B, L, D_y),
                    'std': base_out['std'].reshape(B, L, D_y),
                }
        x = batch[0]
        encoding = self._encoder(x)
        hidden = self._memory_unit(encoding)[0]
        corr = (self._decoder(torch.cat([encoding, hidden], dim=-1))
                * self._corr_max_magnitude)
        # TODO: Maybe we also want to output some rescaling constant here?
        return {'corr': corr, 'mean': base_out['mean'], 'std': base_out['std']}

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
        std = net_out['std']
        corr = net_out['corr']
        y, mask = batch[1:]
        num_valid = mask[:, 1:].sum().item()
        diffs = y - mean
        sq_diffs = diffs.pow(2)
        norm_sq_diffs = sq_diffs / std.pow(2)
        corr_term = 1 - corr[:, 1:].pow(2)
        loss = ((
            (2 * corr_term).pow(-1) * (
                norm_sq_diffs[:, :-1]
                + norm_sq_diffs[:, 1:]
                - (2 * corr[:, 1:] * diffs[:, :-1] * diffs[:, 1:]
                   / (std[:, 1:] * std[:, :-1]))
            )
            + 0.5 * corr_term.log()
        ) * mask[:, 1:]).mean(dim=-1).sum() / num_valid
        stats = {}
        stats['corr_loss'] = loss.item()
        stats['correlation/mean'] = ((corr[:, 1:] * mask[:, 1:]).mean(dim=-1).sum()
                                     / num_valid).item()
        stats['correlation/min'] = (corr[:, 1:] * mask[:, 1:]).min().item()
        stats['correlation/max'] = (corr[:, 1:] * mask[:, 1:]).max().item()
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
            if self.uq_rnn_type == 'gru':
                self._hidden_state = torch.zeros(self._num_layers, net_in.shape[0],
                                                 self._hidden_size, device=self.device)
            else:
                self._hidden_state = tuple(
                    torch.zeros(self._num_layers, net_in.shape[0],
                                self._hidden_size, device=self.device)
                    for _ in range(2))
        else:
            tocompare = (self._hidden_state if self.uq_rnn_type == 'gru'
                         else self._hidden_state[0])
            if tocompare.shape[1] != net_in.shape[0]:
                raise ValueError('Number of inputs does not match previously given '
                                 f'number. Expected {tocompare.shape[1]} but received'
                                 f' {net_in.shape[0]}.')
        # Estimate a Gaussian from the base model.
        _, info = self._base_model.multi_sample_output_from_torch(net_in)
        mean_predictions = info['mean_predictions']
        std_predictions = info['std_predictions']
        # Get the correlation.
        with torch.no_grad():
            encoded = self._encoder(net_in).unsqueeze(1)
            mem_out, hidden_out = self._memory_unit(encoded, self._hidden_state)
            if self._record_history:
                self._hidden_state = hidden_out
            corr_predictions = (
                self._decoder(torch.cat([encoded, mem_out], dim=-1)).squeeze(1)
                * self._corr_max_magnitude
            )
        # If this is not the first prediction adjust the sample based on the corr.
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

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure the optimizer"""
        return (
            torch.optim.AdamW(
                self._base_model.parameters(),
                lr=self._base_learning_rate,
                weight_decay=self._base_weight_decay,
            ),
            torch.optim.AdamW(
                list(self._encoder.parameters())
                + list(self._decoder.parameters())
                + list(self._memory_unit.parameters()),
                lr=self._uq_learning_rate,
                weight_decay=self._uq_weight_decay,
            ),
        )

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
        return 0

    def clear_history(self) -> None:
        """Clear the history."""
        self._hidden_state = None
        self._last_pred_info = None

    def reset(self) -> None:
        """Reset the dynamics model."""
        self.clear_history()
        self._base_model.reset()
