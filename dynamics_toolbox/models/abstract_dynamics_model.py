"""
Abstract class for all dynamics models.

Author: Ian Char
"""
import abc
from typing import Dict, Tuple, Any

import numpy as np


class AbstractDynamicsModel(metaclass=abc.ABCMeta):
    """Abstract model for predicting next states in dynamics."""

    def reset(self) -> None:
        """Reset the dynamics model."""
        pass

    @abc.abstractmethod
    def predict(
            self,
            states: np.ndarray,
            actions: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Predict the next state given current state and action.

        Args:
            states: The current states as a torch tensor.
            actions: The actions to be played as a torch tensor.

        Returns:
            The model output and give a dictionary of related quantities.
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
