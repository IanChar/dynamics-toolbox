"""
Catboost classifier model.

Author: Youngseog Chung
"""
from catboost import CatBoostClassifier

from typing import Sequence, Tuple, Dict, Any, Optional, Callable

from dynamics_toolbox.models.catboost_models.abstract_catboost_model import AbstractCatboostModel

class CatboostRegressor(AbstractCatboostModel):
    """ Catboost classifier. """

    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            learning_rate: float = 1e-3,
            depth: Optional[int] = None,
            loss_type: str = losses.CE,
            weight_decay: Optional[float] = 0.0,
            **kwargs,
    ):
        """Constructor.

        Args:
            input_dim: The input dimension.
            output_dim: The output dimension, equal to number of classes
            learning_rate: The learning rate for the network.
            depth: The number of .
            layer_size: The size of each hidden layer in the MLP.
            architecture: The architecture of the MLP described as a
                a string of underscore separated ints e.g. 256_100_64.
                If provided, this overrides num_layers and layer_sizes.
            hidden_activation: Activation to use.
            loss_type: The name of the loss function to use.
            weight_decay: Goes into l2_leaf_reg
            class_weights

        """
        super().__init__(input_dim, output_dim, **kwargs)
        self.save_hyperparameters()
        hidden_sizes = get_architecture(num_layers, layer_size, architecture)
        self._net = FCNetwork(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_sizes=hidden_sizes,
            hidden_activation=get_activation(hidden_activation),
        )
        self._learning_rate = learning_rate
        self._weight_decay = weight_decay
        self._sample_mode = ''
        self._loss_function = get_classification_loss(loss_type)
        self._loss_type = loss_type
        # TODO: In the future we may want to pass this in as an argument.
        self._metrics = {
            # 'EV': ExplainedVariance(),
            # 'IndvEV': ExplainedVariance('raw_values'),

        }
        ### YSC
        self._hard_labels = bool(kwargs.get('hard_labels', True))
