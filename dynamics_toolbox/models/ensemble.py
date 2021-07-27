"""
A dynamics model that is an ensemble of other dynamics models.

Author: Ian Char
"""
from typing import Sequence, Tuple, Dict, Any, NoReturn, Optional, List

import numpy as np

from dynamics_toolbox.models.abstract_dynamics_model import AbstractDynamicsModel
from dynamics_toolbox.constants import sampling_modes


class FiniteEnsemble(AbstractDynamicsModel):

    def __init__(
            self,
            members: List[AbstractDynamicsModel],
            sample_mode: str = sampling_modes.SAMPLE_MEMBER_EVERY_TRAJECTORY,
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
                raise ValueError('All members must have the same input dimension but found '
                                 f'{self._input_dim} and {member.input_dim}.')
            if member.output_dim != self._output_dim:
                raise ValueError('All members must have the same output dimension but found '
                                 f'{self._output_dim} and {member.output_dim}.')
        self._members = members
        self._sample_mode = sample_mode
        self._nxt_member_to_sample = np.random.randint(len(members))
        if elite_idxs is None:
            self._elite_idxs = list(range(len(self.members)))

    def reset(self) -> NoReturn:
        """Reset the model."""
        self._nxt_member_to_sample = np.random.choice(self._elite_idxs)

    def predict(self, states: np.ndarray, actions: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Predict the next state given current state and action.
        Args:
            states: The current states as a torch tensor.
            actions: The actions to be played as a torch tensor.
        Returns: The next state and give a dictionary of related quantities.
        """
        info_dict = {}
        nxts = []
        for member_idx, member in enumerate(self.members):
            nxt, info = member.predict(states, actions)
            nxts.append(nxt)
            info_dict.update({f'member{member_idx}_{k}': v for k, v in info.items()})
        nxts = np.array(nxts)
        if self._sample_mode == sampling_modes.RETURN_MEAN:
            return np.mean(nxts, axis=0), info_dict
        elif self._sample_mode == sampling_modes.SAMPLE_MEMBER_EVERY_STEP:
            return nxts[np.random.choice(self._elite_idxs)], info_dict
        elif self._sample_mode == sampling_modes.SAMPLE_MEMBER_EVERY_TRAJECTORY:
            return nxts[self._nxt_member_to_sample], info_dict

    def set_member_sample_mode(self, mode: str) -> NoReturn:
        """Set the sampling mode of each member of the ensemble.
        Args:
            mode: The sampling mode.
        """
        for member in self.members:
            self._member.sample_mode = mode

    @property
    def sample_mode(self) -> str:
        """The sample mode is the method that in which we get next state."""
        return self._sample_mode

    @sample_mode.setter
    def sample_mode(self, mode: str) -> NoReturn:
        """Set the sample mode to the appropriate mode."""
        self._sample_mode = mode

    @property
    def members(self) -> Sequence[AbstractDynamicsModel]:
        """The members of the ensemble."""
        return self._members

    @property
    def elite_idxs(self) -> List[int]:
        """Return the indexes of the elite members."""
        return self._elite_idxs

    @elite_idxs.setter
    def elite_idxs(self, idxs: List[int]) -> NoReturn:
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
