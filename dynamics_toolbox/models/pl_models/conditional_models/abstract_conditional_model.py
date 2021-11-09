"""
Abstract class for models that can be conditioned.

Author: Ian Char
Date: 11/7/2021
"""
import abc
from typing import Any

import torch

from dynamics_toolbox.models.pl_models.abstract_pl_model import AbstractPlModel


class AbstractConditionalModel(AbstractPlModel, metaclass=abc.ABCMeta):
    """Models that can be conditioned"""

    @abc.abstractmethod
    def clear_condition(self) -> None:
        """Clear the latent posterior and set back to the prior."""

    @abc.abstractmethod
    def condition_samples(
            self,
            conditions_x: torch.Tensor,
            conditions_y: torch.Tensor,
    ) -> Any:
        """Set the latent posterior of the neural process based on data observed.

        Args:
            conditions_x: The x points to condition on.
            conditions_y: The corresponding y points to condition on.

        Returns:
            The latent encoding of the model.
        """
