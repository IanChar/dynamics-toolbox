"""
MLP but instead of learning a single paramter, learn simplex of parameters.

Model based on https://arxiv.org/abs/2102.10472

Author: Ian Char
"""
from typing import Sequence, Tuple, Dict, Any, Optional, Callable

import torch
from torchmetrics import ExplainedVariance

from dynamics_toolbox.constants import sampling_modes
from dynamics_toolbox.models.pl_models.abstract_pl_model import AbstractPlModel
import dynamics_toolbox.constants.activations as activations
import dynamics_toolbox.constants.losses as losses
from dynamics_toolbox.utils.misc import s2i
from dynamics_toolbox.utils.pytorch.activations import get_activation
from dynamics_toolbox.utils.pytorch.losses import get_regression_loss
from dynamics_toolbox.utils.pytorch.fc_network import FCNetwork


class SimplexMLP(AbstractPlModel):
    """Fully connected network where simplex of weights are low loss."""

    def __init__(
            self,
            num_vertices: int,
            input_dim: int,
            output_dim: int,
            learning_rate: float,
            diversity_coef: float,
            num_layers: Optional[int] = None,
            layer_size: Optional[int] = None,
            architecture: Optional[str] = None,
            hidden_activation: str = activations.RELU,
            loss_type: str = losses.MSE,
            sample_mode: str = sampling_modes.SAMPLE_FROM_DIST,
            weight_decay: Optional[float] = 0.0,
            **kwargs,
    ):
        """Constructor.

        Args:
            num_vertices: The number of vertices that make up the simplex.
            input_dim: The input dimension.
            output_dim: The output dimension.
            learning_rate: The learning rate for the network.
            diversity_coef: The coefficient for encouraging diversity.
            num_layers: The number of hidden layers in the MLP.
            layer_size: The size of each hidden layer in the MLP.
            architecture: The architecture of the MLP described as a
                a string of underscore separated ints e.g. 256_100_64.
                If provided, this overrides num_layers and layer_sizes.
            hidden_activation: Activation to use.
            loss_type: The name of the loss function to use.
            sample_mode: The type of sampling to perform.
            weight_decay: The weight decay for the optimizer.
        """

        super().__init__()
        self.save_hyperparameters()
        if architecture is not None:
            hidden_sizes = s2i(architecture)
        elif num_layers is not None and layer_size is not None:
            hidden_sizes = [layer_size for _ in range(num_layers)]
        else:
            raise ValueError(
                'MLP architecture not provided. Either specify architecture '
                'argument or both num_layers and layer_size arguments.'
            )
        self._num_vertices = num_vertices
        for idx in range(num_vertices):
            setattr(self, f'vertex_{idx}', FCNetwork(
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_sizes=hidden_sizes,
                hidden_activation=get_activation(hidden_activation),
            ))
        self._learning_rate = learning_rate
        self._weight_decay = weight_decay
        self._diversity_coef = diversity_coef
        self._loss_function = get_regression_loss(loss_type)
        self._loss_type = loss_type
        self._sample_mode = sample_mode
        self._curr_sample = None
        self._simplex_dist = torch.distributions.dirichlet.Dirichlet(
            torch.ones(num_vertices))
        # TODO: In the future we may want to pass this in as an argument.
        self._metrics = {
            'EV': ExplainedVariance(),
            'IndvEV': ExplainedVariance('raw_values'),
        }

    def reset(self) -> None:
        """Reset the dynamics model."""
        self._curr_sample = None

    def forward(
            self,
            x: torch.Tensor,
            weighting: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward function for network

        Args:
            x: The input to the network.
            weighting: The point in the simplex to use to sample. Should have
                shape (x.shape[0], num_vertices).

        Returns:
            The output of the network.
        """
        if weighting is None:
            weighting = self._simplex_dist.sample((x.shape[0],))
        n_layers = getattr(self, 'vertex_0').n_layers
        hidden_activation = getattr(self, 'vertex_0').hidden_activation
        out_activation = getattr(self, 'vertex_0').out_activation
        curr = x
        for layer_num in range(n_layers - 1):
            lin_outs = torch.stack([self._get_vertex_layer(v, layer_num)(curr)
                                    for v in range(self._num_vertices)])
            curr = torch.mul(lin_outs.T, weighting).sum(dim=-1).T
            curr = hidden_activation(curr)
        lin_outs = torch.stack([self._get_vertex_layer(v, n_layers - 1)(curr)
                                for v in range(self._num_vertices)])
        curr = torch.mul(lin_outs.T, weighting).sum(dim=-1).T
        if out_activation is not None:
            return out_activation(curr)
        return curr

    def multi_sample_model_from_torch(
            self,
            net_in: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Get the next state as a delta in state.

        It is assumed that each input of the network should be drawn from a different
        sample.

        Args:
            net_in: The input for the network.

        Returns:
            The deltas for next states and dictionary of info.
        """
        if (self._sample_mode == sampling_modes.SAMPLE_MEMBER_EVERY_STEP
            or self._curr_sample is None):
            self._curr_sample = self._simplex_dist.sample((len(net_in),))
        elif len(self._curr_sample) < len(net_in):
            self._curr_sample = torch.cat(
                [self._curr_sample,
                 self._simplex_dist.sample((len(net_in) - len(self._curr_sample),))],
                dim=0)
        weight = self._curr_sample[:len(net_in)]
        with torch.no_grad():
            deltas = self.forward(net_in, weight)
        info = {'delta': deltas}
        return deltas, info

    def get_net_out(self, batch: Sequence[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Get the output of the network and organize into dictionary.

        Args:
            batch: The batch passed to the network.

        Returns:
            Dictionary of name to tensor.
        """
        xi, _ = batch
        output = self.forward(xi)
        return {'prediction': output}

    def loss(
            self,
            net_out: Dict[str, torch.Tensor],
            batch: Sequence[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute the loss function.

        Args:
            net_out: The output of the network.
            batch: The batch passed into the network.

        Returns:
            The loss and a dictionary of other statistics.
        """
        _, yi = batch
        loss = self._loss_function(net_out['prediction'], yi)
        similarity = self._get_cosine_similarity()
        loss += self._diversity_coef * similarity
        stats = {'loss': loss.item(), 'cosine_similarity': similarity.item()}
        return loss, stats

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure the optimizer"""
        return torch.optim.Adam(self.parameters(), lr=self._learning_rate)

    @property
    def sample_mode(self) -> str:
        """The sample mode is the method that in which we get next state."""
        return self._sample_mode

    @sample_mode.setter
    def sample_mode(self, mode: str) -> None:
        """Set the sample mode to the appropriate mode."""
        self._sample_mode = mode

    @property
    def input_dim(self) -> int:
        """The sample mode is the method that in which we get next state."""
        return self._input_dim

    @property
    def output_dim(self) -> int:
        """The sample mode is the method that in which we get next state."""
        return self._output_dim

    @property
    def metrics(self) -> Dict[str, Callable[[torch.Tensor], torch.Tensor]]:
        """Get the list of metric functions to compute."""
        return self._metrics

    @property
    def learning_rate(self) -> float:
        """Get the learning rate."""
        return self._learning_rate

    @property
    def weight_decay(self) -> float:
        """Get the weight decay."""
        return self._weight_decay

    def _get_cosine_similarity(
            self,
            weightings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Get cosine similarity for regularization.

        Code taken from https://github.com/apple/learning-subspaces/blob/9e4cdcf4cb928
        35f8e66d5ed13dc01efae548f67/trainers/train_one_dim_subspaces.py

        Args:
            weightings: The two weightings to compare should be shape (2, num_verts).

        Result:
            The cosine similarity.
        """
        if weightings is None:
            weightings = self._simplex_dist.sample((2,))
        n_layers = getattr(self, 'vertex_0').n_layers
        num = 0.0
        normi = 0.0
        normj = 0.0
        for k in range(n_layers):
            vi = self._get_interior_layer(weightings[0], k)
            vj = self._get_interior_layer(weightings[1], k)
            num += (vi * vj).sum()
            normi += vi.pow(2).sum()
            normj += vj.pow(2).sum()
        return num.pow(2) / (normi * normj)

    def _get_vertex_layer(self, vert_num: int, layer_num: int) -> torch.nn.Linear:
        """Get a vertex layer.

        Args:
            vert_num: The index of the vertex.
            layer_num: The index of the layer.

        Returns:
            The specified layer.
        """
        return getattr(getattr(self, f'vertex_{vert_num}'), f'linear_{layer_num}')

    def _get_interior_layer(
            self,
            weighting: torch.Tensor,
            layer_num: int,
    ) -> torch.Tensor:
        """Get the layer of a non-vertex point on the simplex.

        Args:
            weighting: The location on the simplex.
            layer_num: The index of the layer.

        Returns:
            The specified layer.
        """
        layer_weights = None
        for vertidx, weight in enumerate(weighting):
            toadd = self._get_vertex_layer(vertidx, layer_num).weight * weight
            if layer_weights is None:
                layer_weights = toadd
            else:
                layer_weights += toadd
        return layer_weights

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
        to_return = {}
        pred = net_out['prediction']
        _, yi = batch
        for metric_name, metric in self._metrics.items():
            metric_value = metric(pred, yi)
            if len(metric_value.shape) > 0:
                for dim_idx, metric_v in enumerate(metric_value):
                    to_return[f'{metric_name}_dim{dim_idx}'] = metric_v
            else:
                to_return[metric_name] = metric_value
        return to_return
