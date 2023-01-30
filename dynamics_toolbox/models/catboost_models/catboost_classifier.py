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
            learning_rate: float = None,
            depth: int = 5,
            loss_function: str = losses.CB_LL,
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
        self._input_dim = input_dim
        self._output_dim = output_dim
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
        self._model.set_params(**param_dict)

    def fit(self, tr_pool, cfg_gpu, **kwargs):
        if cfg_gpu is not None:
            self.set_additional_model_params({'task_type': 'GPU', 'devices': cfg_gpu})

        return self._model.fit(tr_pool, **kwargs)

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

    def get_model_out(self, batch: Sequence[np.ndarray]) -> Dict[str, np.ndarray]:
        xi, _ = batch
        output = self.model.pred_prob(xi)
        return {'prediction': output}

    def loss(
            self,
            net_out: Dict[str, np.ndarray],
            batch: Sequence[np.ndarray],
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        pass

    @property
    def metrics(self) -> Dict[str, Callable[[np.ndarray], np.ndarray]]:
        pass

    @property
    def learning_rate(self) -> float:
        """Get the learning rate."""
        return self._learning_rate

    @property
    def weight_decay(self) -> float:
        """Get the weight decay."""
        return self._weight_decay

    @property
    def input_dim(self) -> int:
        """The sample mode is the method that in which we get next state."""
        return self._input_dim

    @property
    def output_dim(self) -> int:
        """The sample mode is the method that in which we get next state."""
        return self._output_dim

    @property
    def sample_mode(self) -> str:
        """The sample mode is the method that in which we get next state."""
        return self._sample_mode
