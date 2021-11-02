"""
Abstract class for models that are train on sequences of SARSA data.

Author: Ian Char
"""
import abc

from dynamics_toolbox.models.pl_models.abstract_pl_model import AbstractPlModel


class AbstractSequentialRlModel(AbstractPlModel, metaclass=abc.ABCMeta):

    @property
    @abc.abstractmethod
    def record_history(self) -> bool:
        """Whether to keep track of the quantities being fed into the neural net."""

    @record_history.setter
    @abc.abstractmethod
    def record_history(self, mode: bool) -> None:
        """Set whether to keep track of quantities being fed into the neural net."""

    def clear_history(self) -> None:
        """Clear the history."""
        pass

    def reset(self) -> None:
        """Reset the dynamics model."""
        self.clear_history()
