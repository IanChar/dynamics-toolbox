"""
Model where the next part in the time series is a linear combination of last
time steps.

Author: Ian Char
Date: March 9, 2023
"""
from typing import Dict, Callable, Tuple, Any, Sequence, Optional

import torch
import torch.nn as nn

from dynamics_toolbox.constants import losses, sampling_modes
from dynamics_toolbox.models.pl_models.sequential_models.abstract_sequential_model \
        import AbstractSequentialModel
from dynamics_toolbox.utils.pytorch.losses import get_regression_loss
from dynamics_toolbox.utils.pytorch.metrics import SequentialExplainedVariance


class LinearAutoregress(AbstractSequentialModel):
    """Linear autoregressive model that also estimates std residuals."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        order: int,
        std_loss_coef: float = 0.1,
        learning_rate: float = 1e-3,
        loss_type: str = losses.MSE,
        sample_mode: str = sampling_modes.SAMPLE_FROM_DIST,
        weight_decay: Optional[float] = 0.0,
        **kwargs,
    ):
        """Constructor.

        Args:
            input_dim: The input dimension.
            output_dim: The output dimension.
            order: The amount of time steps to look back.
            std_loss_coef: Coefficient to weight std estimation in total loss.
        """
        assert input_dim == output_dim
        assert order >= 1
        super().__init__(input_dim, output_dim, **kwargs)
        self.order = order
        self.linear = nn.Linear(order, 1)
        self.stds = nn.Parameter(torch.zeros(output_dim))
        self.std_loss_coef = std_loss_coef
        self._loss_function = get_regression_loss(loss_type)
        self._loss_type = loss_type
        self._hidden_state = None
        self._weight_decay = weight_decay
        self._sample_mode = sample_mode
        self._record_history = True
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
        assert batch[0].shape[1] >= self.order,\
            (f'Need sequence of length at least {self.order}. '
             f'Received {batch[0].shape[1]}.')
        preds = self.linear(batch[0].unfold(1, self.order, 1))
        return {'prediction': preds}

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
        y, mask = batch[1:]
        assert y.shape[1] >= self.order and mask.shape[1] >= mask
        y, mask = [x[:, self.order:] for x in (y, mask)]
        pred_loss = self._loss_function(net_out['prediction'] * mask, y * mask)
        resids = (net_out['prediction'].detach() * mask - y * mask)
        std_resids = resids.view(-1, resids.shape[-1]).std(dim=0)
        std_loss = self._loss_function(self.stds, std_resids)
        loss = pred_loss + self.std_loss_coef * std_loss
        stats = {
            'loss': loss.item(),
            'pred_loss': pred_loss.item(),
            'std_loss': std_loss.item(),
        }
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
            self._hidden_state = net_in.unsqueeze(1)
        elif self._hidden_state.shape[0] != net_in.shape[0]:
            raise ValueError('Number of inputs does not match previously given '
                             f'number. Expected {self._hidden_state.shape[0]} but'
                             f' received {net_in.shape[0]}.')
        lin_in = torch.cat([
            self._hidden_state[:, -(self.order + 1):],
            net_in.unsqueeze(1),
        ], dim=1)
        if self._record_history:
            self._hidden_state = lin_in
        if self._hidden_state[1] < self.order:
            mean_pred = torch.zeros(*lin_in.shape, device=self.device)
        else:
            with torch.no_grad():
                mean_pred = self.linear(lin_in)
        if self._sample_mode == sampling_modes.SAMPLE_FROM_DIST:
            preds = (mean_pred
                     + torch.randn_like(mean_pred) * self.stds.detach().view(1, -1))
        else:
            preds = mean_pred
        info = {
            'predictions': preds,
            'mean_predictions': mean_pred,
            'std_predictions': self.std.detach(),
        }
        return preds, info

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
