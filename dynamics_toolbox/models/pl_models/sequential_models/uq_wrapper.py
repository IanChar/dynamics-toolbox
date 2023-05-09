"""
Model that wraps separate model and learns additional UQ over it. This UQ may help w

1. Calibration.
2. Correlated Errors.

Author: Ian Char
Date: April 27, 2023
"""
from typing import Any, Dict, Callable, Tuple, Sequence, Optional

import numpy as np
from scipy.stats import chi2
import torch
from torch.distributions.normal import Normal
import uncertainty_toolbox as uct

from dynamics_toolbox.constants import sampling_modes
from dynamics_toolbox.models.pl_models.sequential_models.abstract_sequential_model \
        import AbstractSequentialModel
from dynamics_toolbox.utils.pytorch.modules.fc_network import FCNetwork
from dynamics_toolbox.metrics.uq_metrics import multivariate_elipsoid_miscalibration


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
        cal_learning_rate: float = 5e-4,
        corr_learning_rate: float = 5e-4,
        corr_max_magnitude: float = 0.99,
        sample_mode: str = sampling_modes.SAMPLE_FROM_DIST,
        cal_weight_decay: Optional[float] = 0.0,
        corr_weight_decay: Optional[float] = 0.0,
        min_cal_coefficient: float = 1e-3,
        max_cal_coefficient: float = 2,
        wrapped_model=None,
        wrapped_model_is_sequential: bool = False,
        quantile_fidelity: int = 50,
        apply_recal: bool = True,
        apply_corr: bool = True,
        learn_first_step_cal_only: bool = False,
        elipsoid_calibration: bool = True,
        **kwargs
    ):
        """Constructor."""
        super().__init__(input_dim, output_dim, **kwargs)
        self._cal_network = FCNetwork(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_sizes=[cal_hidden_layer_size for _ in range(cal_num_hidden_layers)],
            out_activation=torch.tanh,
        )
        if corr_encoder_num_hidden_layers > 0:
            encoder_hidden_sizes = [corr_encoder_hidden_size
                                    for _ in range(corr_encoder_num_hidden_layers)]
        else:
            encoder_hidden_sizes = None
        decoder_hidden_sizes = [corr_decoder_hidden_size
                                for _ in range(corr_decoder_num_hidden_layers)]
        self._encoder = FCNetwork(
            input_dim=input_dim * 2,
            output_dim=corr_encode_dim,
            hidden_sizes=encoder_hidden_sizes,
        )
        self._difference_bn = torch.nn.BatchNorm1d(input_dim)
        self.corr_rnn_type = corr_rnn_type.lower()
        if corr_rnn_type.lower() == 'gru':
            rnn_class = torch.nn.GRU
        elif corr_rnn_type.lower() == 'lstm':
            rnn_class = torch.nn.LSTM
        else:
            rnn_class = None
        if rnn_class is not None:
            self._memory_unit = rnn_class(corr_encode_dim, corr_rnn_hidden_size,
                                          num_layers=corr_rnn_num_layers,
                                          batch_first=True)
        else:
            self._memory_unit = None
            corr_rnn_hidden_size = 0
        self._decoder = FCNetwork(
            input_dim=corr_encode_dim + corr_rnn_hidden_size,
            output_dim=output_dim,
            hidden_sizes=decoder_hidden_sizes,
            out_activation=torch.tanh,
        )
        self._min_cal_coefficient = min_cal_coefficient
        self._max_cal_coefficient = max_cal_coefficient
        self._wrapped_model = wrapped_model
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._wrapped_is_seq = wrapped_model_is_sequential
        self._encode_dim = corr_encode_dim
        self._hidden_size = corr_rnn_hidden_size
        self._num_layers = corr_rnn_num_layers
        self._cal_learning_rate = cal_learning_rate
        self._corr_learning_rate = corr_learning_rate
        self._cal_weight_decay = cal_weight_decay
        self._corr_weight_decay = corr_weight_decay
        self._sample_mode = sample_mode
        self._corr_max_magnitude = corr_max_magnitude
        self._apply_recal = apply_recal
        self._apply_corr = apply_corr
        self._learn_first_step_cal_only = learn_first_step_cal_only
        self._elipsoid_calibration = elipsoid_calibration
        self._record_history = True
        self._hidden_state = None
        self._last_pred_info = None
        self._last_in = None
        exp_proportions = torch.linspace(0.05, 0.95, quantile_fidelity)
        residual_thresholds = torch.Tensor(chi2(self._output_dim).ppf(
                exp_proportions.numpy()))
        residual_thresholds = residual_thresholds.reshape(1, 1, -1)
        exp_proportions = exp_proportions.reshape(1, -1)
        upper_thresholds = Normal(0, 1).icdf(0.5 * (1 + exp_proportions))
        lower_thresholds = Normal(0, 1).icdf(0.5 * (1 - exp_proportions))
        self.register_buffer('_exp_proportions', exp_proportions)
        self.register_buffer('_residual_thresholds', residual_thresholds)
        self.register_buffer('_upper_thresholds', upper_thresholds)
        self.register_buffer('_lower_thresholds', lower_thresholds)

    def training_step(
            self,
            batch: Sequence[torch.Tensor],
            batch_idx: int,
            optimizer_idx: int,
    ) -> torch.Tensor:
        """Training step for pytorch lightning. Returns the loss."""
        batch = [self.normalizer.normalize(b, 0) if bidx == 0 else b
                 for bidx, b in enumerate(batch)]
        net_out = self.get_net_out(
            batch,
            get_corrs=bool(optimizer_idx),
            suppress_cal_grads=bool(optimizer_idx),
        )
        if optimizer_idx:
            loss, loss_dict = self.corr_loss(net_out, batch)
        else:
            loss, loss_dict = self.cal_loss(net_out, batch)
        self._log_stats(loss_dict, prefix='train')
        return loss

    def validation_step(
        self,
        batch: Sequence[torch.Tensor],
        batch_idx: int,
    ) -> None:
        """Training step for pytorch lightning. Returns the loss."""
        batch = [self.normalizer.normalize(b, 0) if bidx == 0 else b
                 for bidx, b in enumerate(batch)]
        net_out = self.get_eval_net_out(batch)
        cal_loss, cal_dict = self.cal_loss(net_out, batch)
        corr_loss, corr_dict = self.corr_loss(net_out, batch)
        corr_dict.update(cal_dict)
        corr_dict.update(self._get_test_and_validation_metrics(net_out, batch))
        self._log_stats(corr_dict, prefix='val')

    def test_step(
        self,
        batch: Sequence[torch.Tensor],
        batch_idx: int,
    ) -> None:
        """Training step for pytorch lightning. Returns the loss."""
        batch = [self.normalizer.normalize(b, 0) if bidx == 0 else b
                 for bidx, b in enumerate(batch)]
        net_out = self.get_eval_net_out(batch)
        cal_loss, cal_dict = self.cal_loss(net_out, batch)
        corr_loss, corr_dict = self.corr_loss(net_out, batch)
        corr_dict.update(cal_dict)
        corr_dict.update(self._get_test_and_validation_metrics(net_out, batch))
        self._log_stats(corr_dict, prefix='test')

    def predict(
            self,
            model_input: np.ndarray,
            each_input_is_different_sample: Optional[bool] = True,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Make predictions using the currently set sampling method.

        Args:
            model_input: The input to be given to the model.
            each_input_is_different_sample: Whether each input should be treated
                as being drawn from a different sample of the model. Note that this
                may not have an effect on all models (e.g. PNN)

        Returns:
            The output of the model and give a dictionary of related quantities.
        """
        if self._wrapped_model is None:
            raise RuntimeError('Need to set wrapped_model before running.')
        model_input = torch.Tensor(model_input).to(self.device)
        uq_normd = self._normalize_prediction_input(model_input)
        wrapped_normd = self._wrapped_model._normalize_prediction_input(model_input)
        if each_input_is_different_sample:
            output, infos = self.multi_sample_output_from_torch(uq_normd,
                                                                wrapped_normd)
        else:
            output, infos = self.single_sample_output_from_torch(uq_normd,
                                                                 wrapped_normd)
        output = self._wrapped_model._unnormalize_prediction_output(output)
        for k, v in infos.items():
            if isinstance(v, torch.Tensor):
                infos[k] = v.detach().cpu().numpy()
        return output.cpu().numpy(), infos

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
                cals = ((self._cal_network(x) + 1) * self._max_cal_coefficient / 2
                        + self._min_cal_coefficient)
        else:
            cals = ((self._cal_network(x) + 1) * self._max_cal_coefficient / 2
                    + self._min_cal_coefficient)
        outs['cals'] = cals
        if get_corrs:
            encoding = self._encoder(torch.cat([
                x[:, 1:],
                self._difference_bn(
                    (x[:, 1:] - x[:, :-1]).transpose(2, 1)).transpose(2, 1),
            ], dim=-1))
            if self._memory_unit is not None:
                hidden = self._memory_unit(encoding)[0]
                encoding = torch.cat([encoding, hidden], dim=-1)
            corr = (self._decoder(encoding)
                    * self._corr_max_magnitude)
            outs['corrs'] = corr
        return outs

    def loss(self, net_out: Dict[str, torch.Tensor],
             batch: Sequence[torch.Tensor]) -> \
            Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Combined correlation and calibration loss.

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
        cal_loss, cal_dict = self.cal_loss(net_out, batch)
        corr_loss, corr_dict = self.corr_loss(net_out, batch)
        corr_dict.update(cal_dict)
        loss = corr_loss / 10 + cal_loss
        corr_dict['loss'] = loss
        return loss, corr_dict

    def corr_loss(self, net_out: Dict[str, torch.Tensor],
                  batch: Sequence[torch.Tensor]) -> \
            Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute the loss function.

        Args:
            net_out: The output of the network.
            batch: The batch passed into the network. This is expected to be a tuple
                * x: (Batch_size, Sequence Length, dim) (Observations and actions)
                * residuals: (Batch_size, Sequence Length, dim)
                * stds: (Batch_size, Sequence Length, dim)

        Returns:
           The loss and a dictionary of other statistics.
        """
        cals, corr = net_out['cals'], net_out['corrs']
        diffs, std = batch[1:]
        sq_diffs = diffs.pow(2)
        norm_sq_diffs = sq_diffs / (std * cals).pow(2)
        corr_term = 1 - corr.pow(2)
        loss = ((norm_sq_diffs[:, :-1]
                 + norm_sq_diffs[:, 1:]
                 - (2 * corr * diffs[:, :-1] * diffs[:, 1:])
                 / (std[:, 1:] * std[:, :-1] * cals[:, 1:] * cals[:, :-1])
                 ) / (2 * corr_term)
                + 0.5 * corr_term.log()
                ).mean()
        stats = {}
        stats['corr_loss'] = loss.item()
        stats['correlation/mean'] = corr.mean().item()
        stats['correlation/std'] = corr.std().item()
        stats['correlation/min'] = corr.min().item()
        stats['correlation/max'] = corr.max().item()
        stats['loss'] = loss.item()
        return loss, stats

    def cal_loss(self, net_out: Dict[str, torch.Tensor],
                 batch: Sequence[torch.Tensor]) -> \
            Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if self._elipsoid_calibration:
            return self.elipsoid_loss(net_out, batch)
        else:
            return self.interval_loss(net_out, batch)

    def interval_loss(self, net_out: Dict[str, torch.Tensor],
                      batch: Sequence[torch.Tensor]) -> \
            Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute the loss function.

        Args:
            net_out: The output of the network.
            batch: The batch passed into the network. This is expected to be a tuple
                * x: (Batch_size, Sequence Length, dim) (Observations and actions)
                * residuals: (Batch_size, Sequence Length, dim)
                * stds: (Batch_size, Sequence Length, dim)

        Returns:
            The loss and a dictionary of other statistics.
        """
        cals = net_out['cals'].unsqueeze(-1)
        residuals, stds = [b.unsqueeze(-1) for b in batch[1:]]
        uppers = cals * stds * self._upper_thresholds
        lowers = cals * stds * self._lower_thresholds
        upper_masks = uppers < residuals
        lower_masks = lowers > residuals
        interval_loss = (
            uppers - lowers
            + 2 / self._exp_proportions * (lowers - residuals) * lower_masks
            + 2 / self._exp_proportions * (residuals - uppers) * upper_masks
        ).mean()
        stats = {}
        stats['cal/std'] = cals.std().item()
        stats['cal/max'] = cals.max().item()
        stats['cal/min'] = cals.min().item()
        stats['interval_loss'] = interval_loss.item()
        stats['loss'] = interval_loss.item()
        return interval_loss, stats

    def elipsoid_loss(self, net_out: Dict[str, torch.Tensor],
                      batch: Sequence[torch.Tensor]) -> \
            Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute the loss function.

        Args:
            net_out: The output of the network.
            batch: The batch passed into the network. This is expected to be a tuple
                * x: (Batch_size, Sequence Length, dim) (Observations and actions)
                * residuals: (Batch_size, Sequence Length, dim)
                * stds: (Batch_size, Sequence Length, dim)

        Returns:
            The loss and a dictionary of other statistics.
        """
        cals = net_out['cals']
        residuals, stds = batch[1:]
        stds = stds * cals
        normd_resids = (residuals / stds).pow(2).sum(dim=-1, keepdim=True)
        soft_threshold = torch.sigmoid(
            10 * (self._residual_thresholds - normd_resids))
        if self._learn_first_step_cal_only:
            obs_props = soft_threshold[:, 0]
        else:
            obs_props = soft_threshold.mean(dim=1)
        calibration_loss = (obs_props - self._exp_proportions).abs().mean()
        stats = {}
        stats['cal/mean'] = cals.mean().item()
        stats['cal/std'] = cals.std().item()
        stats['cal/max'] = cals.max().item()
        stats['cal/min'] = cals.min().item()
        stats['calibration_loss'] = calibration_loss.item()
        stats['loss'] = calibration_loss.item()
        return calibration_loss, stats

    def single_sample_output_from_torch(
        self,
        uq_in: torch.Tensor,
        wrapped_in: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Get the output for a single sample in the model.

        Args:
            uq_in: The input for the UQ wrapper.
            wrapped_in: The input for the wrapped model.

        Returns:
            The predictions for a single function sample.
        """
        if self._wrapped_model is None:
            raise RuntimeError('Need to set wrapped_model before running.')
        if self._hidden_state is None:
            if self.corr_rnn_type == 'gru':
                self._hidden_state = torch.zeros(self._num_layers, uq_in.shape[0],
                                                 self._hidden_size, device=self.device)
            else:
                self._hidden_state = tuple(
                    torch.zeros(self._num_layers, uq_in.shape[0],
                                self._hidden_size, device=self.device)
                    for _ in range(2))
        else:
            tocompare = (self._hidden_state if self.corr_rnn_type == 'gru'
                         else self._hidden_state[0])
            if tocompare.shape[1] != uq_in.shape[0]:
                raise ValueError('Number of inputs does not match previously given '
                                 f'number. Expected {tocompare.shape[1]} but received'
                                 f' {uq_in.shape[0]}.')
        # Estimate a Gaussian from the base model.
        _, info = self._wrapped_model.multi_sample_output_from_torch(wrapped_in)
        mean_predictions, std_predictions = self._handle_mixture_model(info)
        # Get the calibration correction.
        with torch.no_grad():
            cals = (self._max_cal_coefficient * (self._cal_network(uq_in) + 1) / 2
                    + self._min_cal_coefficient)
        if self._apply_recal:
            std_predictions = std_predictions * cals
        # Get the correlation.
        if self._last_in is None:
            diff = torch.zeros(uq_in.shape)
            diff.to(self.device)
        else:
            diff = self._difference_bn(
                    (uq_in - self._last_in).unsqueeze(-1)).squeeze(-1)
        with torch.no_grad():
            encoded = self._encoder(torch.cat([
                uq_in,
                diff
            ], dim=-1)).unsqueeze(1)
            if self._memory_unit is not None:
                mem_out, hidden_out = self._memory_unit(encoded, self._hidden_state)
                if self._record_history:
                    self._hidden_state = hidden_out
                encoded = torch.cat([encoded, mem_out], dim=-1)
            corr_predictions = (
                self._decoder(encoded).squeeze(1)
                * self._corr_max_magnitude
            )
        # If this is not the first prediction adjust the sample based on the corr.
        if self._sample_mode == sampling_modes.SAMPLE_FROM_DIST:
            if self._last_pred_info is not None and self._apply_corr:
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
        self._last_in = uq_in
        info = {'predictions': predictions,
                'mean_predictions': mean_predictions,
                'std_predictions': std_predictions}
        return predictions, info

    def multi_sample_output_from_torch(
        self,
        uq_in: torch.Tensor,
        wrapped_in: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Get the output where each input is assumed to be from a different sample.

        Args:
            uq_in: The input for the UQ wrapper.
            wrapped_in: The input for the wrapped model.

        Returns:
            The deltas for next states and dictionary of info.
        """
        return self.single_sample_output_from_torch(uq_in, wrapped_in)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure the optimizer"""
        corr_parameters = (
            list(self._encoder.parameters())
            + list(self._decoder.parameters())
            + list(self._difference_bn.parameters())
        )
        if self._memory_unit is not None:
            corr_parameters = corr_parameters + list(self._memory_unit.parameters())
        return (
            torch.optim.AdamW(
                self._cal_network.parameters(),
                lr=self._cal_learning_rate,
                weight_decay=self._cal_weight_decay,
            ),
            torch.optim.AdamW(
                corr_parameters,
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
        mixing_term = 2 / (members ** 2) * torch.sum(torch.cat([torch.stack([
                means[i] * means[j]
                for j in range(i)])
            for i in range(1, members)], dim=0), dim=0)
        std_out = torch.sqrt(mean_var + mean_sq - mixing_term)
        return mean_out, std_out

    def _get_test_and_validation_metrics(
            self,
            net_out: Dict[str, torch.Tensor],
            batch: Sequence[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Compute additional metrics to be used for validation/test only.
        The main one to compute here is the hard ellipsoid calibration and average
        single dimensional interval calibration.

        Args:
            net_out: The output of the network.
            batch: The batch passed into the network.

        Returns:
            A dictionary of additional metrics.
        """
        to_return = {}
        cals = net_out['cals']
        residuals, stds = batch[1:]
        cals, residuals, stds = [vect.reshape(-1, vect.shape[-1]).cpu().numpy()
                                 for vect in (cals, residuals, stds)]
        recalled = cals * stds
        # Elipsoid calibration.
        el_cal, el_over = multivariate_elipsoid_miscalibration(
            means=np.zeros(stds.shape),
            stds=recalled,
            truths=residuals,
            include_overconfidence_scores=True
        )
        to_return['ellipsoid_miscalibration'] = el_cal
        to_return['ellipsoid_overconfidence'] = el_over
        # Single dimension interval calibration.
        interval_miscals = [uct.mean_absolute_calibration_error(
            y_pred=np.zeros(len(recalled)),
            y_std=recalled[:, d],
            y_true=residuals[:, d],
        ) for d in range(recalled.shape[-1])]
        to_return['interval_miscallibration/avg'] = np.mean(interval_miscals)
        to_return['interval_miscallibration/std'] = np.std(interval_miscals)
        to_return['interval_miscallibration/min'] = np.min(interval_miscals)
        to_return['interval_miscallibration/max'] = np.max(interval_miscals)
        return to_return

    @property
    def apply_recal(self):
        return self._apply_recal

    @apply_recal.setter
    def apply_recal(self, mode):
        self._apply_recal = mode

    @property
    def apply_corr(self):
        return self._apply_corr

    @apply_corr.setter
    def apply_corr(self, mode):
        self._apply_corr = mode

    @property
    def wrapped_model(self):
        return self._wrapped_model

    def set_wrapped_model(self, model):
        self._wrapped_model = model

    @property
    def wrapped_is_seq(self):
        return self._wrapped_is_seq

    @wrapped_is_seq.setter
    def wrapped_is_seq(self, is_seq):
        self._wrapped_is_seq = is_seq

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
        self._wrapped_model.reset()
        self._last_in = None
