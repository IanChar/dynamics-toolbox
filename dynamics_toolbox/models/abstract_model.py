"""
Abstract class for all dynamics models.

Author: Ian Char
"""
import abc
from typing import Dict, Tuple, Any, Optional

import numpy as np


class AbstractModel(metaclass=abc.ABCMeta):
    """Abstract model for predicting next states in dynamics."""

    def reset(self) -> None:
        """Reset the model."""
        pass

    @abc.abstractmethod
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

    @property
    @abc.abstractmethod
    def sample_mode(self) -> str:
        """The sample mode is the method that in which we get next state."""

    @sample_mode.setter
    @abc.abstractmethod
    def sample_mode(self, mode: str) -> None:
        """Set the sample mode to the appropriate mode."""

    @property
    @abc.abstractmethod
    def input_dim(self) -> int:
        """The sample mode is the method that in which we get next state."""

    @property
    @abc.abstractmethod
    def output_dim(self) -> int:
        """The sample mode is the method that in which we get next state."""
