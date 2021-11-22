"""
Recurrent network that returns Gaussian distribution of next point.

Author: Ian Char
Date: 11/4/2021
"""
from typing import Dict, Callable, Tuple, Any, Sequence, Optional

import hydra.utils
import torch
from omegaconf import DictConfig
from torchmetrics import ExplainedVariance

from dynamics_toolbox.constants import losses, sampling_modes
from dynamics_toolbox.models.pl_models import PNN
from dynamics_toolbox.models.pl_models.sequential_models.abstract_sequential_model import \
    AbstractSequentialModel
from dynamics_toolbox.utils.pytorch.losses import get_regression_loss


class DynammicsRPNN(AbstractSequentialModel):
    """Recurrent network that outputs gaussian distribution."""

    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            encode_dim: int,
            num_layers: int,
            hidden_size: int,
            encoder_cfg: DictConfig,
            pnn_decoder_cfg: DictConfig,
            warm_up_period: int = 0,
            learning_rate: float = 1e-3,
            loss_type: str = losses.MSE,
            weight_decay: Optional[float] = 0.0,
            sample_mode: str = sampling_modes.SAMPLE_FROM_DIST,
            **kwargs,
    ):
        """Constructor.

        Args:
            input_dim: The input dimension.
            output_dim: The output dimension.
            encode_dim: The dimension of the encoder output.
            num_layers: Number of layers in the memory unit.
            hidden_size: The number hidden units in the memory unit.
            encoder_cfg: The configuration for the encoder network.
            pnn_decoder_cfg: The configuration for the decoder network. This must be
                a PNN.
            warm_up_period: The amount of data to take in before predictions begin to
                be made.
            learning_rate: The learning rate for the network.
            loss_type: The name of the loss function to use.
            weight_decay: The weight decay for the optimizer.
            sample_mode: The method to use for sampling.
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
            input_dim=encode_dim,
            output_dim=output_dim,
            _recursive_=False,
        )
        assert isinstance(self._decoder, PNN), 'Decoder must be a PNN.'
        self._memory_unit = torch.nn.GRU(encode_dim, hidden_size,
                                         num_layers=num_layers,
                                         device=self.device)
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._encode_dim = encode_dim
        self._hidden_size = hidden_size
        self._num_layers = num_layers
        self._warm_up_period = warm_up_period
        self._learning_rate = learning_rate
        self._weight_decay = weight_decay
        self._sample_mode = sample_mode
        self._decoder.sample_mode = sample_mode
        self._loss_function = get_regression_loss(loss_type)
        self._loss_type = loss_type
        self._record_history = True
        self._hidden_state = None
        # TODO: In the future we may want to pass this in as an argument.
        self._metrics = {
            'EV': ExplainedVariance(),
            'IndvEV': ExplainedVariance('raw_values'),
        }

    def get_net_out(self, batch: Sequence[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Get the output of the network and organize into dictionary.

        Args:
            batch: The batch passed into the network. This is expected to be a tuple
                with (obs, acts, nxts, rews, terminals).

        Returns:
            Dictionary of name to tensor.
        """
        assert len(batch) == 6, 'Need SARS + terminal + is_real in batch.'
        obs, acts = batch[:2]
        is_real = batch[-1]
        if len(obs.shape) == 2:
            obs = obs.unsqueeze(1)
            acts = acts.unsqueeze(1)
        mean_predictions, logvar_predictions = [], []
        hidden = torch.zeros(self._num_layers, obs.shape[0], self._encode_dim,
                             device=self.device)
        curr = obs[:, 0, :]
        for t in range(obs.shape[1]):
            net_in = torch.cat([curr, acts[:, t]], dim=1)
            encoded = self._encoder(net_in)
            mem_out, hidden = self._memory_unit(encoded.unsqueeze(0), hidden)
            mean_pred, logvar_pred = self._decoder(mem_out.squeeze(0))
            mean_pred *= is_real[:, t].unsqueeze(-1)
            logvar_pred *= is_real[:, t].unsqueeze(-1)
            mean_predictions.append(mean_pred)
            logvar_predictions.append(logvar_pred)
            if t < self._warm_up_period:
                curr = obs[:, t + 1, :]
            else:
                curr = (curr + mean_pred + torch.randn_like(curr).to(self.device)
                        * (logvar_pred * 0.5).exp())
        return {
            'mean': torch.stack(mean_predictions, dim=1),
            'logvar': torch.stack(logvar_predictions, dim=1),
        }

    def loss(self, net_out: Dict[str, torch.Tensor], batch: Sequence[torch.Tensor]) -> \
            Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute the loss function.

        Args:
            net_out: The output of the network.
            batch: The batch passed into the network. This is expected to be a tuple with
                (obs, acts, nxts, rews, terminals).

        Returns:
            The loss and a dictionary of other statistics.
        """
        _, _, nxts, _, _ = batch
        return self._decoder.loss(
            {'mean': net_out['mean'][:, self._warm_up_period:, :],
             'logvar': net_out['logvar'][:, self._warm_up_period:, :]},
            [None, nxts[:, self._warm_up_period:, :]]
        )

    def single_sample_output_from_torch(self, net_in: torch.Tensor) -> Tuple[
        torch.Tensor, Dict[str, Any]]:
        """Get the output for a single sample in the model.

        Args:
            net_in: The input for the network.

        Returns:
            The predictions for a single function sample.
        """
        if self._hidden_state is None:
            self._hidden_state = torch.zeros(self._num_layers, net_in.shape[0],
                                             self._encode_dim, device=self.device)
        elif self._hidden_state.shape[1] != net_in.shape[0]:
            raise ValueError('Number of inputs does not match previously given number.'
                             f' Expected {self._hidden_state.shape[1]} but received'
                             f' {net_in.shape[0]}.')
        with torch.no_grad():
            encoded = self._encoder(net_in)
            mem_out, hidden_out = self._memory_unit(encoded.unsqueeze(0),
                                                    self._hidden_state)
            if self._record_history:
                self._hidden_state = hidden_out
        return self._decoder.single_sample_output_from_torch(mem_out.squeeze(0))

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
    def sample_mode(self) -> str:
        """The sample mode is the method that in which we get next state."""
        return self._sample_mode

    @sample_mode.setter
    def sample_mode(self, mode: str) -> None:
        """Set the sample mode to the appropriate mode."""
        self._sample_mode = mode
        self._decoder.sample_mode = mode

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

    def _get_test_and_validation_metrics(
            self,
            net_out: Dict[str, torch.Tensor],
            batch: Sequence[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Compute additional metrics to be used for validation/test only.

        Args:
            net_out: The output of the network.
            batch: The batch passed into the network.

        Returns:
            A dictionary of additional metrics.
        """
        return super()._get_test_and_validation_metrics(
            {'prediction': net_out['mean']},
            batch,
        )
