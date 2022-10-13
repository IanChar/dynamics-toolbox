"""
A recursive network for fusion.

Author: Ian Char
Date: August 11, 2022
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
from dynamics_toolbox.utils.pytorch.modules.transformer import *
import random

class DynamicsTransformer(AbstractSequentialModel):
    """Transformer network."""

    def __init__(
            self,
            input_dim: int,
            output_dim: int,

            num_hidden: int, 
            ff_dim: int, 
            embed_dim: int,
            num_heads: int, 
            teacher_force=True,

            warm_up_period: int = 0,
            learning_rate: float = 1e-3,
            loss_type: str = losses.MSE,
            weight_decay: Optional[float] = 0.0,
            autoregress_noise: Optional[float] = 0.0,
            seq_len = 0,  
            dropout_prob=0.3,
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
        """
        super().__init__(input_dim, output_dim, **kwargs)
        self.save_hyperparameters()

        # these will now be transformers,
        # have to rewrite the configs

       # self.encoder = hydra.utils.instantiate(
        #    input_dim=input_dim,
         #   output_dim=encode_dim,
         #   _recursive_=False,
        #)

        self.encoder = TransformerEncoder(num_hidden = num_hidden,
                                         input_dim=input_dim, 
                                         embed_dim = embed_dim,
                                         ff_dim = ff_dim, 
                                         output_dim = output_dim,
                                         num_heads = num_heads, 
                                         dropout_prob = dropout_prob,
                                         seq_len = 30
                                         )

        self.transformer = TransformerForPrediction(self.encoder)


        # have to rewrite this part to match the useful params
        self._predictions_are_deltas = predictions_are_deltas
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
        # TODO: In the future we may want to pass this in as an argument.
        self._metrics = {
            'EV': ExplainedVariance(),
            'IndvEV': ExplainedVariance('raw_values'),
        }

    def get_net_out(self, batch: Sequence[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Get the output of the network and organize into dictionary.

        Args:
            batch: The batch passed into the network. This is expected to be a tuple
                with (state, actuator, next_actuator, next_state, is_padding).

        Returns:
            Dictionary of name to tensor.
        """
        assert len(batch) == 6, 'Invalide batch size.'
        assert len(batch) == 6, 'Need SARS + terminal + is_real in batch.'

        obs, acts = batch[:2]
        is_real = batch[-1]
        if len(obs.shape) == 2:
            obs = obs.unsqueeze(1)
            acts = acts.unsqueeze(1)
        predictions = []
        curr = obs[:, 0, :]
        memory = torch.cat([curr, acts[:, 0]], dim=1)
        memory = memory.unsqueeze(1) # batch size, 1, number of dimensions
        curr = obs[:, 0, :]

        # batch size by time steps by number of states
        memory = torch.cat([curr, acts[:, 0]], dim=1)
        memory = memory.unsqueeze(1) # batch size, 1, number of dimensions
        
        for t in range(obs.shape[1]):

            pred = self.transformer(memory)
            predictions.append(pred * is_real[:, t].unsqueeze(-1))
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

            if t<obs.shape[1]-1:
                next_input = torch.cat([curr, acts[:, t+1]], dim=1)
                memory = torch.cat([memory, next_input.unsqueeze(1)], dim=1)


        return {'prediction': torch.stack(predictions, dim=1)}



    def loss(self, net_out: Dict[str, torch.Tensor], batch: Sequence[torch.Tensor]) -> \
            Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute the loss function.

        Args:
            net_out: The output of the network.
            batch: The batch passed into the network. This is expected to be a tuple with
                (state, actuator, next_state, next_actuator is_padding).

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
        if self._hidden_state is None:
            self._hidden_state = torch.zeros(self._num_layers, net_in.shape[0],
                                             self._encode_dim, device=self.device)
        elif self._hidden_state.shape[1] != net_in.shape[0]:
            raise ValueError('Number of inputs does not match previously given number.'
                             f' Expected {self._hidden_state.shape[1]} but received'
                             f' {net_in.shape[0]}.')
        with torch.no_grad():
            encoded = self.transformer(net_in)
            mem_out, hidden_out = self._memory_unit(encoded.unsqueeze(0),
                                                     self._hidden_state)
            if self._record_history:
                self._hidden_state = hidden_out
            predictions = self._decoder(mem_out.squeeze(0))
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

    def reset(self) -> None:
        """Reset the dynamics model."""
        self.clear_history()
