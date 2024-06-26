"""
A dynamics model that is an ensemble of other dynamics models.

Author: Ian Char
"""
from collections import defaultdict
from typing import Sequence, Tuple, Dict, Any, Optional, List

import numpy as np
import torch

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
            elite_idxs: The indexes of the elite members. These are the members to
                sample from.
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
        if hasattr(self._members[0], 'reset'):
            for member in self._members:
                member.reset()

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
        info_dict = defaultdict(list)
        nxts = []
        for member_idx, member in enumerate(self.members):
            nxt, info = member.predict(
                model_input,
                each_input_is_different_sample=each_input_is_different_sample,
            )
            nxts.append(nxt)
            for k, v in info.items():
                info_dict[k].append(v)
        nxts = np.array(nxts)
        info_dict = {k: np.array(v) for k, v in info_dict.items()}
        num_membs = len(self.members)
        if self._sample_mode == sampling_modes.RETURN_MEAN:
            return np.mean(nxts, axis=0), info_dict
        elif self._sample_mode == sampling_modes.BOOTSTRAP_EST_INDEPENDENT_DIM:
            mean = np.mean(nxts, axis=0)
            stderr = np.std(nxts, axis=0) / np.sqrt(len(self.members))
            info_dict['ensemble_mean'] = mean
            info_dict['ensemble_stderr'] = stderr
            return mean + np.random.standard_normal(mean.shape) * stderr, info_dict
        elif self._sample_mode == sampling_modes.BOOTSTRAP_EST_JOINT_DIM:
            mean = np.mean(nxts, axis=0)
            info_dict['ensemble_mean'] = mean
            normd_nxts = (nxts - mean[np.newaxis]).transpose(axes=[1, 0, 2])
            covs = (normd_nxts @ normd_nxts.transpose(axes=[0, 2, 1]) /
                    (num_membs * np.sqrt(num_membs)))
            return np.vstack([
                np.random.multivariate_normal(mu, sigma).reshape(1, -1)
                for mu, sigma in zip(mean, covs)]), info_dict
        else:
            if (self._curr_sample is None
                    or self._sample_mode == sampling_modes.SAMPLE_MEMBER_EVERY_STEP):
                self._curr_sample = self.draw_from_categorical(len(model_input))
            elif len(model_input) > len(self._curr_sample):
                self._curr_sample = np.vstack([
                    self._curr_sample,
                    self.draw_from_categorical(
                        len(model_input) - len(self._curr_sample)),
                ])
            if each_input_is_different_sample:
                ensemble_idxs = self._curr_sample[:len(model_input)]
            else:
                ensemble_idxs = [self._curr_sample[0] for _ in range(len(model_input))]
            return nxts[ensemble_idxs, np.arange(len(model_input))], info_dict

    def single_sample_output_from_torch(
            self,
            net_in: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Get the output for a single sample in the model.

        Args:
            net_in: The input for the network.

        Returns:
            The deltas for next states and dictionary of info.
        """
        if self._sample_mode == sampling_modes.SAMPLE_MEMBER_EVERY_STEP:
            self.reset()
        # if self._efficient_sampling:
        #     return self._efficient_forward(net_in, single_sample=True)
        if self._curr_sample is None:
            self._curr_sample = self.draw_from_categorical(len(net_in))
        info_dict = defaultdict(list)
        deltas = []
        for member_idx, member in enumerate(self.members):
            delta, info = member.single_sample_output_from_torch(net_in)
            deltas.append(delta)
            for k, v in info.items():
                info_dict[k].append(v)
        deltas = torch.stack(deltas)
        for k, v in info_dict.items():
            info_dict[k] = torch.stack(v)
        samp_idxs = self._curr_sample[0].repeat(len(net_in))
        sampled_delta = deltas[samp_idxs, torch.arange(len(net_in))]
        info_dict['deltas'] = deltas
        return sampled_delta, info_dict

    def multi_sample_output_from_torch(
            self,
            net_in: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Get the output where each input is assumed to be from a different sample.

        Args:
            net_in: The input for the network.

        Returns:
            The deltas for next states and dictionary of info.
        """
        if self._sample_mode == sampling_modes.SAMPLE_MEMBER_EVERY_STEP:
            self.reset()
        # if self._efficient_sampling:
        #     return self._efficient_forward(net_in, single_sample=False)
        if self._curr_sample is None:
            self._curr_sample = self.draw_from_categorical(len(net_in))
        elif len(self._curr_sample) < len(net_in):
            self._curr_sample = torch.cat([
                self._curr_sample,
                self.draw_from_categorical(len(net_in) - len(self._curr_sample)),
            ], dim=0)
        info_dict = defaultdict(list)
        deltas = []
        for member in self.members:
            delta, info = member.multi_sample_output_from_torch(net_in)
            deltas.append(delta)
            for k, v in info.items():
                info_dict[k].append(v)
        deltas = torch.stack(deltas)
        for k, v in info_dict.items():
            info_dict[k] = torch.stack(v)
        samp_idxs = self._curr_sample[:len(net_in)]
        sampled_delta = deltas[samp_idxs, torch.arange(len(net_in))]
        info_dict['deltas'] = deltas
        return sampled_delta, info_dict

    def set_sample(self, sample: np.ndarray) -> None:
        """Set the current sample.

        Args:
            sample: Ndarray of the indices to use.
        """
        self._curr_sample = sample

    def set_member_sample_mode(self, mode: str) -> None:
        """Set the sampling mode of each member of the ensemble.

        Args:
            mode: The sampling mode.
        """
        for member in self.members:
            member.sample_mode = mode

    def to(self, device):
        self._members = [m.to(device) for m in self._members]
        return self

    def _normalize_prediction_input(self, model_input: torch.Tensor) -> torch.Tensor:
        """Normalize the input for prediction.

        Args:
            model_input: The input to the model.

        Returns:
            The normalized input.
        """
        return self.members[0]._normalize_prediction_input(model_input)

    def _unnormalize_prediction_output(self, output: torch.Tensor) -> torch.Tensor:
        """Unnormalize the output of the model.

        Args:
            output: The output of the model.

        Returns:
            The unnormalized outptu.
        """
        return self.members[0]._unnormalize_prediction_output(output)

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

    @property
    def normalizer(self):
        return self.members[0].normalizer

    def draw_from_categorical(self, num_samples) -> np.ndarray:
        """Draw from categorical distribution.

        Args:
            num_samples: The number of samples to draw.

        Returns:
            The draws from the distribution.
        """
        return np.random.randint(len(self._members), size=(num_samples,))
