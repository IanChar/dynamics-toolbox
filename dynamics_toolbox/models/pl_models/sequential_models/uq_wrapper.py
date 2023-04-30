"""
Model that wraps separate model and learns additional UQ over it. This UQ may help w

1. Calibration.
2. Correlated Errors.

Author: Ian Char
Date: April 27, 2023
"""
from typing import Any, Dict, Callable, Tuple, Sequence, Optional

import torch

from dynamics_toolbox.constants import sampling_modes
from dynamics_toolbox.models.pl_models.sequential_models.abstract_sequential_model \
        import AbstractSequentialModel
from dynamics_toolbox.utils.pytorch.modules.fc_network import FCNetwork


class UQWrapper(AbstractSequentialModel):

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        cal_num_hidden_layers: int,
        cal_hidden_layer_size: int,
        corr_encode_dim: int,
        corr_rnn_hidden_size: int,
        corr_decoder_num_hidden_layers: int,
        corr_decoder_hidden_size: int,
        corr_encoder_num_hidden_layers: int = 0,
        corr_encoder_hidden_size: int = 0,
        corr_rnn_num_layers: int = 1,
        corr_rnn_type: str = 'gru',
        cal_learning_rate: float = 1e-3,
        corr_learning_rate: float = 5e-4,
        corr_max_magnitude: float = 0.9,
        sample_mode: str = sampling_modes.SAMPLE_FROM_DIST,
        cal_weight_decay: Optional[float] = 0.0,
        corr_weight_decay: Optional[float] = 0.0,
        min_cal_coefficient: float = 1e-3,
        base_model=None,
        base_model_is_sequential: bool = False,
        **kwargs
    ):
        """Constructor."""
        super().__init__(input_dim, output_dim, **kwargs)
        self._cal_network = FCNetwork(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_sizes=[cal_hidden_layer_size for _ in range(cal_num_hidden_layers)],
            out_activation=torch.nn.functional.softmax,
        )
        if corr_encoder_num_hidden_layers > 0:
            encoder_hidden_sizes = [corr_encoder_hidden_size
                                    for _ in range(corr_encoder_num_hidden_layers)]
        else:
            encoder_hidden_sizes = None
        decoder_hidden_sizes = [corr_decoder_hidden_size
                                for _ in range(corr_decoder_num_hidden_layers)]
        self._encoder = FCNetwork(
            input_dim=input_dim,
            output_dim=corr_encode_dim,
            hidden_sizes=encoder_hidden_sizes,
        )
        self._decoder = FCNetwork(
            input_dim=corr_encode_dim + corr_rnn_hidden_size,
            output_dim=output_dim,
            hidden_sizes=decoder_hidden_sizes,
            out_activation=torch.tanh,
        )
        self.corr_rnn_type = corr_rnn_type.lower()
        if corr_rnn_type.lower() == 'gru':
            rnn_class = torch.nn.GRU
        elif corr_rnn_type.lower() == 'lstm':
            rnn_class = torch.nn.LSTM
        else:
            raise ValueError(f'Cannot recognize RNN type {corr_rnn_type}')
        self._memory_unit = rnn_class(corr_encode_dim, corr_rnn_hidden_size,
                                      num_layers=corr_rnn_num_layers,
                                      batch_first=True)
        self._min_cal_coefficient = min_cal_coefficient
        self._base_model = base_model
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._base_is_seq = base_model_is_sequential
        self._encode_dim = corr_encode_dim
        self._hidden_size = corr_rnn_hidden_size
        self._num_layers = corr_rnn_num_layers
        self._cal_learning_rate = cal_learning_rate
        self._corr_learning_rate = corr_learning_rate
        self._cal_weight_decay = cal_weight_decay
        self._corr_weight_decay = corr_weight_decay
        self._sample_mode = sample_mode
        self._corr_max_magnitude = corr_max_magnitude
        self._record_history = True
        self._hidden_state = None
        self._last_pred_info = None

    def training_step(
            self,
            batch: Sequence[torch.Tensor],
            batch_idx: int,
            optimizer_idx: int,
    ) -> torch.Tensor:
        """Training step for pytorch lightning. Returns the loss."""
        batch = self.normalizer.normalize_batch(batch)
        net_out = self.get_net_out(
            batch,
            get_corrs=bool(optimizer_idx),
            suppress_cal_grads=bool(optimizer_idx),
        )
        if optimizer_idx:
            loss, loss_dict = self.corr_loss(net_out, batch)
        else:
            loss, loss_dict = self.cal_loss(net_out, batch)
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
        net_out = self.get_eval_net_out(batch)
        cal_loss, cal_dict = self.cal_loss(net_out, batch)
        self._log_stats(cal_dict, prefix='val')
        corr_loss, corr_dict = self.cal_loss(net_out, batch)
        self._log_stats(corr_dict, prefix='val')

    def test_step(
        self,
        batch: Sequence[torch.Tensor],
        batch_idx: int,
    ) -> None:
        """Training step for pytorch lightning. Returns the loss."""
        batch = self.normalizer.normalize_batch(batch)
        net_out = self.get_eval_net_out(batch)
        cal_loss, cal_dict = self.cal_loss(net_out, batch)
        self._log_stats(cal_dict, prefix='test')
        corr_loss, corr_dict = self.cal_loss(net_out, batch)
        self._log_stats(corr_dict, prefix='test')

    def get_net_out(
        self,
        batch: Sequence[torch.Tensor],
        get_corrs: bool = True,
        suppress_cal_grads: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Get the output of the network and organize into dictionary.

        Args:
            batch: The batch passed into the network. This is expected to be a tuple
                * x: (Batch_size, Sequence Length, dim) (Observations and actions)
                * means: (Batch_size, Sequence Length, dim)
                * stds: (Batch_size, Sequence Length, dim)
                * residuals: (Batch_size, Sequence Length, dim)

        Returns:
            Dictionary of name to tensor.
                * mean (batch_size, Sequence Length, dim)
                * logvar (batch_size, Sequence Length, dim)
                * corr (batch_size, Sequence Length, dim)
        """
        x = batch[0]
        B, L, D = x.shape
        # Forward pass of base model.
        # TODO: This happens twice per step bc of pl. Is there a way of caching?
        outs = {}
        if suppress_cal_grads:
            with torch.no_grad():
                cals = self._cal_network(x) + self._min_cal_coefficient
        else:
            cals = self._cal_network(x) + self._min_cal_coefficient
        outs['cals'] = cals
        if get_corrs:
            encoding = self._encoder(x)
            hidden = self._memory_unit(encoding)[0]
            corr = (self._decoder(torch.cat([encoding, hidden], dim=-1))
                    * self._corr_max_magnitude)
            outs['corrs'] = corr
        return outs

    def corr_loss(self, net_out: Dict[str, torch.Tensor],
                  batch: Sequence[torch.Tensor]) -> \
            Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute the loss function.

        Args:
            net_out: The output of the network.
            batch: The batch passed into the network. This is expected to be a tuple
                * x: (Batch_size, Sequence Length, dim) (Observations and actions)
                * means: (Batch_size, Sequence Length, dim)
                * stds: (Batch_size, Sequence Length, dim)
                * residuals: (Batch_size, Sequence Length, dim)

        Returns:
            The loss and a dictionary of other statistics.
        """
        cals, corr = net_out['cals'], net_out['corrs']
        std, diffs = batch[2:]
        sq_diffs = diffs.pow(2)
        norm_sq_diffs = sq_diffs / (std * cals).pow(2)
        corr_term = 1 - corr[:, 1:].pow(2)
        loss = ((
            (2 * corr_term).pow(-1) * (
                norm_sq_diffs[:, :-1]
                + norm_sq_diffs[:, 1:]
                - (2 * corr[:, 1:] * diffs[:, :-1] * diffs[:, 1:]
                   / (std[:, 1:] * std[:, :-1] * cals[:, 1:] * cals[:, :-1]))
            )
            + 0.5 * corr_term.log()
        )).mean()
        stats = {}
        stats['corr_loss'] = loss.item()
        stats['correlation/mean'] = corr[:, 1:].mean().item()
        stats['correlation/min'] = corr[:, 1:].min().item()
        stats['correlation/max'] = corr[:, 1:].min().item()
        stats['loss'] = loss.item()
        return loss, stats

    def cal_loss(self, net_out: Dict[str, torch.Tensor],
                 batch: Sequence[torch.Tensor]) -> \
            Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute the loss function.

        Args:
            net_out: The output of the network.
            batch: The batch passed into the network. This is expected to be a tuple
                * x: (Batch_size, Sequence Length, dim) (Observations and actions)
                * means: (Batch_size, Sequence Length, dim)
                * stds: (Batch_size, Sequence Length, dim)
                * residuals: (Batch_size, Sequence Length, dim)

        Returns:
            The loss and a dictionary of other statistics.
        """
        cals = net_out['cals'].unsqueeze(-1)
        stds, residuals = [b.unsqueeze(-1) for b in batch[2:]]
        uppers = cals * stds * self._upper_quantiles
        lowers = cals * stds * self._upper_quantiles
        upper_masks = uppers < residuals
        lower_masks = lowers > residuals
        interval_loss = (
            uppers - lowers
            + 2 / self._coverages * (lowers - residuals) * lower_masks
            + 2 / self._coverages * (residuals - uppers) * upper_masks
        ).mean()
        stats = {}
        stats['cal/mean'] = cals.mean().item()
        stats['cal/std'] = cals.std().item()
        stats['inteval_loss'] = interval_loss.item()
        stats['loss'] = interval_loss.item()
        return interval_loss, stats

    def single_sample_output_from_torch(self, net_in: torch.Tensor) -> Tuple[
            torch.Tensor, Dict[str, Any]]:
        """Get the output for a single sample in the model.

        Args:
            net_in: The input for the network with expected shape (batch size, dim)

        Returns:
            The predictions for a single function sample.
        """
        if self._base_model is None:
            raise RuntimeError('Need to set base_model before running.')
        if self._hidden_state is None:
            if self.corr_rnn_type == 'gru':
                self._hidden_state = torch.zeros(self._num_layers, net_in.shape[0],
                                                 self._hidden_size, device=self.device)
            else:
                self._hidden_state = tuple(
                    torch.zeros(self._num_layers, net_in.shape[0],
                                self._hidden_size, device=self.device)
                    for _ in range(2))
        else:
            tocompare = (self._hidden_state if self.corr_rnn_type == 'gru'
                         else self._hidden_state[0])
            if tocompare.shape[1] != net_in.shape[0]:
                raise ValueError('Number of inputs does not match previously given '
                                 f'number. Expected {tocompare.shape[1]} but received'
                                 f' {net_in.shape[0]}.')
        # Estimate a Gaussian from the base model.
        _, info = self._base_model.multi_sample_output_from_torch(net_in)
        mean_predictions, std_predictions = self._handle_mixture_model(info)
        # Get the calibration correction.
        with torch.no_grad():
            cals = self._cal_network(net_in)
        std_predictions = std_predictions * cals
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
                self._cal_network.parameters(),
                lr=self._cal_learning_rate,
                weight_decay=self._cal_weight_decay,
            ),
            torch.optim.AdamW(
                list(self._encoder.parameters())
                + list(self._decoder.parameters())
                + list(self._memory_unit.parameters()),
                lr=self._corr_learning_rate,
                weight_decay=self._corr_weight_decay,
            ),
        )

    def _handle_mixture_model(self, info: Dict[str, torch.Tensor]):
        """If we get multiple predictions assume we are dealing with a Gaussian
           mixture model and handle accordingly.
        """
        if len(info['mean_predictions'].shape) == 2:
            return info['mean_predictions'], info['std_predictions']
        means, stds = info['mean_predictions'], info['std_predictions']
        members = len(means)
        mean_out = torch.mean(means, dim=0)
        mean_var = torch.mean(stds.pow(2), dim=0)
        mean_sq = torch.mean(means.pow(2), dim=0) * (1 - 1 / members)
        mixing_term = 2 / (members ** 2) * torch.sum(torch.cat([torch.Tensor([
                means[i] * means[j]
                for j in range(i)])
            for i in range(1, members)]), dim=0)
        std_out = torch.sqrt(mean_var + mean_sq - mixing_term)
        return mean_out, std_out

    @property
    def base_model(self):
        return self._base_model

    @base_model.setter
    def base_model(self, model):
        self._base_model = model

    @property
    def base_is_seq(self):
        return self._base_is_seq

    @base_is_seq.setter
    def base_is_seq(self, is_seq):
        self._base_is_seq = is_seq

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
