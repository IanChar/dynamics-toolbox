"""
Abstract misc model for learning dynamics.

Author: Youngseog Chung
"""
import abc
from typing import Dict, Tuple, Sequence, Any, Callable, Optional

import numpy as np
import torch
from pytorch_lightning import LightningModule

from dynamics_toolbox.models.abstract_model import AbstractModel
from dynamics_toolbox.utils.pytorch.modules.normalizer import Normalizer, NoNormalizer


class AbstractCatboostModel(AbstractModel, metaclass=abc.ABCMeta):
    """Abstract model for predicting next states in dynamics."""

    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            normalizer: Optional[Normalizer] = None,
            **kwargs
        ):
        """Constructor.

        Args:
            input_dim: The input dimension.
            output_dim: The output dimension.
            normalizer: Normalizer for the model.
        """
        super().__init__()
        self._input_dim = input_dim
        self._output_dim = output_dim
        if normalizer is None:
            normalizer = NoNormalizer()
        self.normalizer = normalizer
        self._normalize_inputs = True
        self._unnormalize_outputs = True

    def training_step():
        # not needed
        pass

    def validation_step():
        # not needed
        pass

    def test_step():
        # not needed
        pass

    def predict(
            self,
            model_input: np.ndarray,
            each_input_is_different_sample: Optional[bool] = True,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Make predictions using the currently set sampling method.

        Args:
            model_input: The input to be given to the model.
            each_input_is_different_sample: Whether each input should be treated
                as being drawn from a different sample of the model. Note that this
                may not have an effect on all models (e.g. PNN)

        Returns:
            The output of the model and give a dictionary of related quantities.
        """
        model_input = self._normalize_prediction_input(model_input)
        if each_input_is_different_sample:
            output, infos = self.multi_sample_output(model_input)
        else:
            output, infos = self.single_sample_output(model_input)
        output = self._unnormalize_prediction_output(output)
        return output, infos

    def configure_optimizers():
        pass


    def get_eval_model_out(self, batch: Sequence[np.ndarray]) -> Dict[str, np.ndarray]:
        """Get the validation output of the network and organize into dictionary.

        Args:
            batch: The batch passed to the network.

        Returns:
            Dictionary of name to tensor.
        """
        return self.get_model_out(batch=batch)

    @property
    def normalize_inputs(self) -> bool:
        """Whether inputs should be normalized when predicting."""
        return self._normalize_inputs

    @normalize_inputs.setter
    def normalize_inputs(self, mode: bool) -> None:
        """Set normalize inputs to true or false."""
        self._normalize_inputs = mode

    @property
    def unnormalize_outputs(self) -> bool:
        """Whether inputs should be normalized when predicting."""
        return self._unnormalize_outputs

    @unnormalize_outputs.setter
    def unnormalize_outputs(self, mode: bool) -> None:
        """Set unnormalize outputs to true or false."""
        self._unnormalize_outputs = mode

    @abc.abstractmethod
    def get_model_out(self, batch: Sequence[np.ndarray]) -> Dict[str, np.ndarray]:
        """Get the output of the network and organize into dictionary.

        Args:
            batch: The batch passed to the network.

        Returns:
            Dictionary of name to tensor.
        """

    @abc.abstractmethod
    def loss(
            self,
            net_out: Dict[str, np.ndarray],
            batch: Sequence[np.ndarray],
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Compute the loss function.

        Args:
            net_out: The output of the network.
            batch: The batch passed into the network.

        Returns:
            The loss and a dictionary of other statistics.
        """

    @abc.abstractmethod
    def single_sample_output(
            self,
            model_in: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Get the output for a single sample in the model.

        Args:
            model_in: The input for the network.

        Returns:
            The deltas for next states and dictionary of info.
        """

    @abc.abstractmethod
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

    @property
    @abc.abstractmethod
    def metrics(self) -> Dict[str, Callable[[np.ndarray], np.ndarray]]:
        """Get the list of metric functions to compute."""

    @property
    @abc.abstractmethod
    def learning_rate(self) -> float:
        """Get the learning rate."""

    @property
    @abc.abstractmethod
    def weight_decay(self) -> float:
        """Get the weight decay."""

    def _normalize_prediction_input(self, model_input: np.ndarray) -> torch.Tensor:
        """Normalize the input for prediction.

        Args:
            model_input: The input to the model.

        Returns:
            The normalized input.
        """
        if self.normalize_inputs:
            return self.normalizer.normalize(model_input, 0)
        return model_input

    def _unnormalize_prediction_output(self, output: torch.Tensor) -> torch.Tensor:
        """Unnormalize the output of the model.

        Args:
            output: The output of the model.

        Returns:
            The unnormalized outptu.
        """
        if self.unnormalize_outputs:
            return self.normalizer.unnormalize(output, 1)
        return output

    def _log_stats(self, *args: Dict[str, float], prefix='train', **kwargs) -> None:
        """Log all of the stats from dictionaries.

        Args:
            args: Dictionaries of torch tensors to add stats about.
            prefix: The prefix to add to the statistic string.
            kwargs: Other kwargs to be passed to self.log.
        """
        for arg in args:
            for stat_name, stat in arg.items():
                self.log(f'{prefix}/{stat_name}', stat, **kwargs)

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
        for metric_name, metric in self.metrics.items():
            metric_value = metric(pred, yi)
            if len(metric_value.shape) > 0:
                for dim_idx, metric_v in enumerate(metric_value):
                    to_return[f'{metric_name}_dim{dim_idx}'] = metric_v
            else:
                to_return[metric_name] = metric_value
        return to_return
