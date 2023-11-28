"""
Recursive model that predicts a gaussian distribtion.

Author: Ian Char
Date: 10/27/2022
"""
from typing import Dict, Callable, Tuple, Any, Sequence, Optional

import hydra.utils
import torch
from omegaconf import DictConfig


from dynamics_toolbox.constants import losses, sampling_modes
from dynamics_toolbox.models.pl_models.sequential_models.abstract_sequential_model \
        import AbstractSequentialModel
from dynamics_toolbox.utils.pytorch.losses import get_regression_loss
from dynamics_toolbox.utils.pytorch.metrics import SequentialExplainedVariance
from dynamics_toolbox.models.pl_models.sequential_models.diffusion_helpers.utils import ddpm_schedules

class DiffusionTransformer(AbstractSequentialModel):
    """Transformer-based Diffusion Model"""

    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            encode_dim: int,
            rnn_num_layers: int,
            rnn_hidden_size: int,
            diffusion: bool,
            diffusion_denoiser_cfg: DictConfig,
            encoder_cfg: DictConfig,
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
        self._encoder = hydra.utils.instantiate(
            encoder_cfg,
            input_dim=input_dim,
            output_dim=encode_dim,
            _recursive_=False,
        )
        self.diffusion = diffusion
        self._diff_model = hydra.utils.instantiate(diffusion_denoiser_cfg, input_dim = encode_dim + rnn_hidden_size, output_dim = output_dim, 
                                                  x_dim = encode_dim + rnn_hidden_size, y_dim = output_dim)
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
        self.ddpm_buffer = {}
        for k, v in ddpm_schedules(self._diff_model.beta1, self._diff_model.beta2, self._diff_model.n_T).items():
            self.ddpm_buffer[k] = v.to(self.device)
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
        """
        
        _ts = torch.randint(1, self._diff_model.n_T + 1, (batch[1].shape[0], batch[1].shape[1], 1)).to(batch[0].device)
        context_mask = torch.bernoulli(torch.zeros(batch[0].shape[1], batch[0].shape[0]) + self._diff_model.drop_prob).to(batch[0].device)

        noise = torch.randn_like(batch[1]).to(batch[0].device)
        y_t = self.ddpm_buffer["sqrtab"].to(batch[1].device)[_ts] *  batch[1] + self.ddpm_buffer["sqrtmab"].to(batch[1].device)[_ts] * noise

        
        encoded = self._encoder(batch[0])
        if self._use_layer_norm:
            encoded = self._layer_norm(encoded)
        mem_out = self._memory_unit(encoded)[0]
        #print("ts, context mask, noise, encoded, mem_out", _ts.shape, context_mask.shape, noise.shape, encoded.shape, mem_out.shape)
        noise_pred_batch = self._diff_model.nn_model(y_t, torch.cat([encoded, mem_out], dim = -1), _ts / self._diff_model.n_T, context_mask)

        return {'noise_pred_batch': noise_pred_batch, 'noise': noise}

    def sample(self, batch: Sequence[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Get the output of the network and organize into dictionary.

        Args:
            batch: The batch passed into the network. This is expected to be a tuple
                * x: (Batch_size, Sequence Length, dim)
                * y: (Batch_size, Sequence Length, dim)
                * mask: (Batch_size, Sequence Length, 1)

        Returns:
            Dictionary of name to tensor.
        """
        is_zero = False
        if self._diff_model.guide_w > -1e-3 and self._diff_model.guide_w < 1e-3:
            is_zero = True   
        
        encoded = self._encoder(batch[0])
        if self._use_layer_norm:
            encoded = self._layer_norm(encoded)
        mem_out = self._memory_unit(encoded)[0]
        x_batch = torch.cat([encoded, mem_out], dim = -1)

        # how many noisy actions to begin with
        n_sample, seq_len = x_batch.shape[0], x_batch.shape[1]
        y_shape = (n_sample, seq_len, self._output_dim)

        # sample initial noise, y_0 ~ N(0, 1),
        y_i = torch.randn(y_shape).to(x_batch.device)

        if not is_zero:
            if len(x_batch.shape) > 2:
                # repeat x_batch twice, so can use guided diffusion
                x_batch = x_batch.repeat(2, 1, 1)
            else:
                # repeat x_batch twice, so can use guided diffusion
                x_batch = x_batch.repeat(2, 1, 1)

            # half of context will be zero
            context_mask = torch.zeros(x_batch.shape[1], x_batch.shape[0]).to(x_batch.device)
            context_mask[:, n_sample:] = 1.0  # makes second half of batch context free
        else:
            context_mask = torch.zeros(x_batch.shape[1], x_batch.shape[0]).to(x_batch.device)


        # run denoising chain
        y_i_store = []  # if want to trace how y_i evolved
        return_y_trace = True
        for i in range(self._diff_model.n_T, 0, -1):
            t_is = torch.tensor([i / self._diff_model.n_T]).to(x_batch.device)
            t_is = t_is.repeat(n_sample, seq_len, 1)

            if not is_zero:
                # double batch
                y_i = y_i.repeat(2, 1, 1)
                t_is = t_is.repeat(2, 1, 1)

            z = torch.randn(y_shape).to(x_batch.device) if i > 1 else 0

            # split predictions and compute weighting
            eps = self._diff_model.nn_model(y_i, x_batch, t_is, context_mask)
            if not is_zero:
                eps1 = eps[:n_sample]
                eps2 = eps[n_sample:]
                eps = (1 + self.guide_w) * eps1 - self.guide_w * eps2
                y_i = y_i[:n_sample]
            y_i = self.ddpm_buffer["oneover_sqrta"].to(x_batch.device)[i] * (y_i - eps * self.ddpm_buffer["mab_over_sqrtmab"].to(x_batch.device)[i]) + self.ddpm_buffer["sqrt_beta_t"].to(x_batch.device)[i] * z
            if return_y_trace and (i % 20 == 0 or i == self._diff_model.n_T or i < 8):
                y_i_store.append(y_i.detach().cpu().numpy())

        if return_y_trace:
            return {'y_i': y_i, 'y_i_store': y_i_store}
        else:
            return {'y_i': y_i}



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

        noise_pred_batch = net_out['noise_pred_batch']
        noise = net_out['noise']
        mask = batch[-1]
        mask[:, :self._warm_up_period, :] = 0
        loss = self._diff_model.loss_mse(noise * mask, noise_pred_batch * mask)
        stats = dict(
            nll=0,
            mse=loss.item(),
        )
        stats['noise_pred/mean'] = (noise_pred_batch * mask).mean().item()
        stats['loss'] = loss.item()
        #print("loss:", stats['loss'])
        return loss, stats

    def loss_eval(self, net_out: Dict[str, torch.Tensor], batch: Sequence[torch.Tensor]) -> \
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
        
        pred = net_out['y_i']
        y, mask = batch[1:]
        mask[:, :self._warm_up_period, :] = 0
        sq_diffs = (pred * mask - y * mask).pow(2)
        mse = torch.mean(sq_diffs)
        loss = mse
        stats = dict(
            mse=mse.item(),
        )
        stats['loss'] = loss.item()
        #print("eval loss", stats['mse'])
        return loss, stats


    def single_sample_output_from_torch(self, net_in: torch.Tensor) -> Tuple[
            torch.Tensor, Dict[str, Any]]:
        """Get the output for a single sample in the model.

        Args:
            net_in: The input for the network with expected shape (batch size, dim)

        Returns:
            The predictions for a single function sample.
        """
        print("here inside single sample output")
        if self._hidden_state is None:
            if self.rnn_type == 'gru':
                self._hidden_state = torch.zeros(self._num_layers, net_in.shape[0],
                                                 self._hidden_size, device=net_in.device)
            else:
                self._hidden_state = tuple(
                    torch.zeros(self._num_layers, net_in.shape[0],
                                self._hidden_size, device=net_in.device)
                    for _ in range(2))
        else:
            tocompare = (self._hidden_state if self.rnn_type == 'gru'
                         else self._hidden_state[0])
            if tocompare.shape[1] != net_in.shape[0]:
                raise ValueError('Number of inputs does not match previously given '
                                 f'number. Expected {topcompare.shape[1]} but received'
                                 f' {net_in.shape[0]}.')
        with torch.no_grad():
            encoded = self._encoder(net_in).unsqueeze(1)
            if self._use_layer_norm:
                encoded = self._layer_norm(encoded)
            mem_out, hidden_out = self._memory_unit(encoded, self._hidden_state)
            if self._record_history:
                self._hidden_state = hidden_out
            
            #encode and memout are of shape (batch size, 1, dim)
            ################## CHANGE FROM HERE ON FOR DIFFUSION MODEL ####################
            print("encoded, mem_out", encoded.shape, mem_out.shape)
            predictions, denoised_pred_trace = self._diff_model.sample(torch.cat([encoded, mem_out], dim=-1), return_y_trace = True)
        info = {'predictions': predictions, 'denoised_pred_trace': denoised_pred_trace}
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

    def reset(self) -> None:
        """Reset the dynamics model."""
        self.clear_history()
