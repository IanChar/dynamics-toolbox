"""
An MLP that is trained in an autorgressive fashion.

Author: Ian Char
Date: 11/22/2021
"""
from typing import Dict, Callable, Tuple, Any, Sequence, Optional

import hydra.utils
import torch
from omegaconf import DictConfig
from torchmetrics import ExplainedVariance

from dynamics_toolbox.constants import losses
from dynamics_toolbox.models.pl_models.sequential_models.abstract_sequential_model import \
    AbstractSequentialModel
from dynamics_toolbox.utils.pytorch.losses import get_regression_loss


class DynamicsMLP(AbstractSequentialModel):
    """RNN network."""

    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            mlp_cfg: DictConfig,
            warm_up_period: int = 0,
            learning_rate: float = 1e-3,
            loss_type: str = losses.MSE,
            weight_decay: Optional[float] = 0.0,
            autoregress_noise: Optional[float] = 0.0,
            predictions_are_deltas: bool = True,
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
            decoder_cfg: The configuration for the decoder network.
            warm_up_period: The amount of data to take in before predictions begin to
                be made.
            learning_rate: The learning rate for the network.
            loss_type: The name of the loss function to use.
            weight_decay: The weight decay for the optimizer.
            autoregress_noise: The amount of noise to apply when feeding predictions
                back in as inputs.
            predictions_are_deltas: Whether the predictions of the model are deltas.
        """
        super().__init__(input_dim, output_dim, **kwargs)
        self.save_hyperparameters()
        self._net = hydra.utils.instantiate(
            mlp_cfg,
            input_dim=input_dim,
            output_dim=output_dim,
            _recursive_=False,
        )
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._warm_up_period = warm_up_period
        self._learning_rate = learning_rate
        self._weight_decay = weight_decay
        self._autoregress_noise = autoregress_noise
        self._sample_mode = ''
        self._loss_function = get_regression_loss(loss_type)
        self._loss_type = loss_type
        self._record_history = True
        self._hidden_state = None
        self._predictions_are_deltas = predictions_are_deltas
        # TODO: In the future we may want to pass this in as an argument.
        self._metrics = {
            'EV': ExplainedVariance(),
            'IndvEV': ExplainedVariance('raw_values'),
        }

    def get_net_out(self, batch: Sequence[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Get the output of the network and organize into dictionary.

        Args:
            batch: The batch passed into the network. This is expected to be a tuple
                with (obs, acts, nxts, rews, terminals, is_padding).

        Returns:
            Dictionary of name to tensor.
        """
        assert len(batch) == 6, 'Need SARS + terminal + is_real in batch.'
        obs, acts = batch[:2]
        is_real = batch[-1]
        if len(obs.shape) == 2:
            obs = obs.unsqueeze(1)
            acts = acts.unsqueeze(1)
        predictions = []
        curr = obs[:, 0, :]
        for t in range(obs.shape[1]):
            net_in = torch.cat([curr.detach(), acts[:, t]], dim=1)
            predictions.append(self._net(net_in) * is_real[:, t].unsqueeze(-1))
            if t < self._warm_up_period:
                curr = obs[:, t + 1, :]
            else:
                if self._predictions_are_deltas:
                    curr = curr + predictions[-1]
                else:
                    curr = predictions[-1]
                if self.training and self._autoregress_noise > 0:
                    curr += (torch.randn_like(curr).to(self.device)
                             * self._autoregress_noise)
        return {'prediction': torch.stack(predictions, dim=1)}

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
        nxts = batch[3]
        loss = self._loss_function(
            net_out['prediction'][:, self._warm_up_period:, :],
            nxts[:, self._warm_up_period:, :],
        )
        stats = {'loss': loss.item()}
        return loss, stats

    def single_sample_output_from_torch(self, net_in: torch.Tensor) -> Tuple[
        torch.Tensor, Dict[str, Any]]:
        """Get the output for a single sample in the model.

        Args:
            net_in: The input for the network.

        Returns:
            The predictions for a single function sample.
        """
        with torch.no_grad():
            predictions = self._encoder(net_in)
        info = {'predictions': predictions}
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
