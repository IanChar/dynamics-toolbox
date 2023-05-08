"""
Schedulers for how many steps in the model should be taken.

Author: Ian Char
Date: April 21, 2023
"""
import abc

import numpy as np


class HorizonScheduler(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def get_horizon(self, num_steps_taken: int) -> int:
        """Get the current horizon.

        Args:
            num_steps_taken: The number of steps taken in the true environment.

        Returns: Returns the horizon for the model.
        """


class ConstantHorizon(HorizonScheduler):

    def __init__(
        self,
        horizon: int,
    ):
        """Constructor.

        Args:
            horizon: The constant horizon to return.
        """
        self._horizon = horizon

    def get_horizon(self, num_steps_taken: int) -> int:
        """Get the current horizon.

        Args:
            num_steps_taken: The number of steps taken in the true environment.

        Returns: Returns the horizon for the model.
        """
        return self._horizon


class LinearlyInterpolateHorizon(HorizonScheduler):

    def __init__(
        self,
        horizon_low: int,
        horizon_high: int,
        slope: float,
    ):
        """Constructor.

        Args:
            horizon_low: The lower bound on the horizon.
            horizon_high: The upper bound on the horizon.
            slope: The slope of the interpolater.
        """
        self._low = horizon_low
        self._high = horizon_high
        self._slope = slope

    def get_horizon(self, num_steps_taken: int) -> int:
        """Get the current horizon.

        Args:
            num_steps_taken: The number of steps taken in the true environment.

        Returns: Returns the horizon for the model.
        """
        return int(np.clip(self._low + num_steps_taken * self._slope,
                           self._low, self._high))
