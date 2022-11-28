"""
Catboost classifier model.

Author: Youngseog Chung
"""
from catboost import CatBoostClassifier
import numpy as np

from typing import Sequence, Tuple, Dict, Any, Optional, Callable

from dynamics_toolbox.models.catboost_models.abstract_catboost_model import AbstractCatboostModel
import dynamics_toolbox.constants.losses as losses

class CBClassifier(AbstractCatboostModel):
    """ Catboost classifier. """

    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            learning_rate: float = 1e-3,
            depth: int = 5,
            loss_function: str = losses.CB_LL,  # TODO: make this
            # loss_type: str = losses.CE,
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
        self._model = CatBoostClassifier(
            depth=depth,
            learning_rate=learning_rate,
            loss_function=loss_function,
            l2_leaf_reg=weight_decay,
        )
        self._loss_function = loss_function
        self._learning_rate = learning_rate
        self._weight_decay = weight_decay
        self._sample_mode = ''
        # TODO: we still need to append the iterations parameter later

        self._metrics = {
            # 'EV': ExplainedVariance(),
            # 'IndvEV': ExplainedVariance('raw_values'),

        }
        # TODO: is hard_labels needed for catboost?
        self._hard_labels = bool(kwargs.get('hard_labels', True))

    def set_additional_model_params(
            self,
            param_dict: Dict[str, any],
    ):
        breakpoint()
        self.model.set_params(*param_dict)


    def single_sample_output(
            self,
            model_in: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Get the output for a single sample in the model.

        Args:
            net_in: The input for the network.

        Returns:
            The predictions for a single function sample
        """
        pred_prob = self.model.predict_proba(model_in)
        pred_class = np.argmax(pred_prob, axis=1)
        info = {
            'pred_prob': pred_prob,
            'pred_class': pred_class,
        }
        return pred_prob, info

    def multi_sample_output(
            self,
            model_in: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Get the output where each input is assumed to be from a different sample.

        Args:
            model_in: The input for the network.

        Returns:
            The deltas for next states and dictionary of info.
        """
        return self.single_sample_output(model_in)
