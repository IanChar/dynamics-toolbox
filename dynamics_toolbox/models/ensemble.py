"""
A dynamics model that is an ensemble of other dynamics models.

Author: Ian Char
"""
from typing import Sequence, Tuple, Dict, Any, Optional, List

import numpy as np

from dynamics_toolbox.models.abstract_model import AbstractModel
from dynamics_toolbox.constants import sampling_modes


class Ensemble(AbstractModel):

    def __init__(
            self,
            members: List[AbstractModel],
            sample_mode: Optional[str] = sampling_modes.SAMPLE_MEMBER_EVERY_TRAJECTORY,
            elite_idxs: Optional[List[int]] = None,
    ):
        """Constructor

        Args:
            members: The members of the ensemble
            sample_mode: The method to use for sampling.
            elite_idxs: The indexes of the elite members. These are the members to sample from.
        """
        self._input_dim = members[0].input_dim
        self._output_dim = members[0].output_dim
        for member in members:
            if member.input_dim != self._input_dim:
                raise ValueError('All members must have the same input dimension but'
                                 f' found {self._input_dim} and {member.input_dim}.')
            if member.output_dim != self._output_dim:
                raise ValueError('All members must have the same output dimension but'
                                 f' found {self._output_dim} and {member.output_dim}.')
        self._members = members
        self._sample_mode = sample_mode
        self._curr_sample = None
        if elite_idxs is None:
            self._elite_idxs = list(range(len(self.members)))

    def reset(self) -> None:
        """Reset the model."""
        self._curr_sample = None

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
        info_dict = {}
        nxts = []
        for member_idx, member in enumerate(self.members):
            nxt, info = member.predict(model_input)
            nxts.append(nxt)
            info_dict.update({f'member{member_idx}_{k}': v for k, v in info.items()})
        nxts = np.array(nxts)
        if self._sample_mode == sampling_modes.RETURN_MEAN:
            return np.mean(nxts, axis=0), info_dict
        else:
            if (self._curr_sample is None
                    or self._sample_mode == sampling_modes.SAMPLE_MEMBER_EVERY_STEP):
                self._curr_sample = self._draw_from_categorical(len(model_input))
            elif len(model_input) > len(self._curr_sample):
                self._curr_sample = np.vstack([
                    self._curr_sample,
                    self._draw_from_categorical(len(model_input) - len(self._curr_sample)),
                ])
            if each_input_is_different_sample:
                ensemble_idxs = self._curr_sample[:len(model_input)]
            else:
                ensemble_idxs = [self._curr_sample[0] for _ in range(len(model_input))]
            return nxts[ensemble_idxs, np.arange(len(model_input))], info_dict

    def set_member_sample_mode(self, mode: str) -> None:
        """Set the sampling mode of each member of the ensemble.

        Args:
            mode: The sampling mode.
        """
        for member in self.members:
            member.sample_mode = mode

    @property
    def sample_mode(self) -> str:
        """The sample mode is the method that in which we get next state."""
        return self._sample_mode

    @sample_mode.setter
    def sample_mode(self, mode: str) -> None:
        """Set the sample mode to the appropriate mode."""
        self._sample_mode = mode

    @property
    def members(self) -> Sequence[AbstractModel]:
        """The members of the ensemble."""
        return self._members

    @property
    def elite_idxs(self) -> List[int]:
        """Return the indexes of the elite members."""
        return self._elite_idxs

    @elite_idxs.setter
    def elite_idxs(self, idxs: List[int]) -> None:
        """
        The elite member indexes.

        Args:
            idxs: The indices of the elite members.
        """
        assert np.all([idx < len(self._members) for idx in idxs])
        self._elite_idxs = idxs

    @property
    def input_dim(self) -> int:
        """The sample mode is the method that in which we get next state."""
        return self.members[0].input_dim

    @property
    def output_dim(self) -> int:
        """The sample mode is the method that in which we get next state."""
        return self.members[0].output_dim

    def _draw_from_categorical(self, num_samples) -> np.ndarray:
        """Draw from categorical distribution.

        Args:
            num_samples: The number of samples to draw.

        Returns:
            The draws from the distribution.
        """
        return np.random.randint(len(self._members), size=(num_samples,))
