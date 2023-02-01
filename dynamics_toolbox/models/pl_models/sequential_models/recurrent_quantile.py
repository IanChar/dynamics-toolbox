"""
A recurrent version of quantile regression.

Author: Ian Char
Date: 11/22/2021
"""
from typing import Optional, Dict, Sequence, Tuple, Any, Callable

import hydra
import torch
from omegaconf import DictConfig

from dynamics_toolbox.constants import sampling_modes
from dynamics_toolbox.models.pl_models import QuantileModel
from dynamics_toolbox.models.pl_models.sequential_models.abstract_sequential_model import \
    AbstractSequentialModel


class RecurrentQuantile(AbstractSequentialModel):

    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            encode_dim: int,
            num_layers: int,
            hidden_size: int,
            encoder_cfg: DictConfig,
            quantile_decoder_cfg: DictConfig,
            warm_up_period: int = 0,
            learning_rate: float = 1e-3,
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
            qauntile_decoder_cfg: The configuration for the decoder network.
                This must be a quantile model.
            warm_up_period: The amount of data to take in before predictions begin to
                be made.
            learning_rate: The learning rate for the network.
            loss_type: The name of the loss function to use.
            weight_decay: The weight decay for the optimizer.
            sample_mode: The method to use for sampling.
        """
        super().__init__(input_dim, output_dim)
        self._encoder = hydra.utils.instantiate(
            encoder_cfg,
            input_dim=input_dim,
            output_dim=encode_dim,
            _recursive_=False,
        )
        self._decoder = hydra.utils.instantiate(
            quantile_decoder_cfg,
            input_dim=encode_dim,
            output_dim=output_dim,
            _recursive_=False,
        )
        assert isinstance(self._decoder, QuantileModel), 'Decoder must be a PNN.'
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
        self._record_history = True
        self._hidden_state = None
        self._eval_q_list = torch.linspace(0.01, 0.99, 99).to(self.device)

    def get_net_out(self, batch: Sequence[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Get the output of the network and organize into dictionary.

        Args:
            batch: The batch passed into the network. This is expected to be
                sequential data.

        Returns:
            Dictionary of name to tensor.
        """
        return self._generic_net_out(batch, self._decoder.get_q_list())

    def get_eval_net_out(self, batch: Sequence[torch.Tensor]) \
            -> Dict[str, torch.Tensor]:
        """Get the validation output of the network and organize into dictionary.

        Args:
            batch: The batch passed into the network. This is expected to be
                sequential data.

        Returns:
            Dictionary of name to tensor.
        """
        return self._generic_net_out(batch, q_list=self._eval_q_list)

    def loss(self, net_out: Dict[str, torch.Tensor], batch: Sequence[torch.Tensor]) -> \
            Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute the loss function.

        Args:
            net_out: The output of the network.
            batch: The batch passed into the network. This is expected to be
                sequential data.

        Returns:
            The loss and a dictionary of other statistics.
        """
        _, yi = batch
        return self._decoder.loss(
            {'q_pred': net_out['q_pred'][:, self._warm_up_period:, :],
             'q_list': net_out['q_list']},
            [None, yi[:, self._warm_up_period:, :]]
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
        return {}

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

    def _generic_net_out(
            self,
            batch: Sequence[torch.Tensor],
            q_list: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Get net out for a generic q_list.

        Args:
            batch: The batch passed into the network. This is expected to be
                sequential data.
            q_list: flat tensor of quantiles.

        Returns:
            Dictionary of name to tensor.
        """
        x_pts = batch[0]
        assert len(x_pts.shape) == 3, 'Data must be sequential.'
        predictions = []
        hidden = torch.zeros(self._num_layers, x_pts.shape[0], self._encode_dim,
                             device=self.device)
        curr = x_pts[:, 0, :]
        for t in range(x_pts.shape[1]):
            encoded = self._encoder(curr)
            mem_out, hidden = self._memory_unit(encoded.unsqueeze(0), hidden)
            predictions.append(self._decoder.forward(mem_out, q_list=q_list))
            if t < x_pts.shape[1]:
                curr = x_pts[:, t + 1, :]
        predictions = torch.stack(predictions, dim=1)
        return {
            'q_pred': torch.stack(predictions, dim=1),
            'q_list': q_list,
        }

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
