"""
Ensemble of other pytorch models.

Author: Ian Char
"""
from collections import defaultdict
from typing import Optional, Sequence, Dict, Callable, Tuple, Any

import hydra.utils
import numpy as np
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
import torch

from dynamics_toolbox.constants import sampling_modes
from dynamics_toolbox.models.pl_models.abstract_pl_model import AbstractPlModel
from dynamics_toolbox.utils.pytorch.modules.normalizer import Normalizer


class SimultaneousEnsemble(AbstractPlModel):
    """Train an ensemble of models simultaneously to minimize loss across all members.

    This class should be mainly used if one wants to train all members at once.
    """

    def __init__(
            self,
            num_members: int,
            member_cfg: DictConfig,
            sample_mode: str = sampling_modes.SAMPLE_MEMBER_EVERY_TRAJECTORY,
            diversity_coef: Optional[float] = 0.0,
            efficient_sampling: bool = False,
            **kwargs,
    ):
        """Constructor

        Args:
            num_members: The number of members in the ensemble.
            member_cfg: The hyperparameters for a member of the ensemble.
            sample_mode: The method to use for sampling.
            diversity_coef: Coefficient of diversity to add to loss.
            efficient_sampling: If this is true then when sampling only do forward
                passes for the networks that are chosen in the ensemble. This will
                not be truly random anymore.
        """
        LightningModule.__init__(self)
        for member_idx in range(num_members):
            setattr(self, f'_member_{member_idx}',
                    hydra.utils.instantiate(member_cfg, _recursive_=False))
        self._num_members = num_members
        self._sample_mode = sample_mode
        self._diversity_coef = diversity_coef
        self._normalize_inputs = True
        self._unnormalize_outputs = True
        self._curr_sample = None
        self._efficient_sampling = efficient_sampling
        # TODO: Could be changed in the future.
        self._similarity = torch.nn.CosineSimilarity(dim=0)

    def reset(self) -> None:
        """Reset the dynamics model."""
        self._curr_sample = None

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Get the optimizers.

        Returns:
            List of the optimizers of the members.
        """
        parameters = []
        for midx in range(self._num_members):
            parameters += list(getattr(self, f'_member_{midx}').parameters())
        return torch.optim.Adam(parameters, lr=self.learning_rate,
                                weight_decay=self.weight_decay)

    def get_net_out(self, batch: Sequence[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Get the output of the network and organize into dictionary.

        Args:
            batch: The batch passed to the network.

        Returns:
            Dictionary of name to tensor.
        """
        to_return = {}
        for member_idx in range(self._num_members):
            member_out = getattr(self, f'_member_{member_idx}').get_net_out(batch)
            for k, v in member_out.items():
                to_return[f'_member_{member_idx}_{k}'] = v
        return to_return

    def loss(
            self,
            net_out: Dict[str, torch.Tensor],
            batch: Sequence[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute the loss function.

        Args:
            net_out: The output of the network.
            batch: The batch passed into the network.

        Returns:
            The loss and a dictionary of other statistics.
        """
        # Parse the net out inputs.
        member_outs = {}
        for k, v in net_out.items():
            splitted = k.split('_')
            member = f'_member_{int(splitted[2])}'
            if member in member_outs:
                member_outs[member][''.join(splitted[3:])] = v
            else:
                member_outs[member] = {''.join(splitted[3:]): v}
        # Compute loss for each member.
        loss_dict = {}
        total_loss = None
        for member, member_out in member_outs.items():
            member_loss, member_loss_dict = getattr(self, member).loss(member_out,
                                                                       batch)
            if total_loss is None:
                total_loss = member_loss / self._num_members
            else:
                total_loss += member_loss / self._num_members
            for k, v in member_loss_dict.items():
                loss_dict[f'{member}_{k}'] = v
        # TODO: Add diversity term to loss.
        if self._diversity_coef > 0:
            member_idx1, member_idx2 = np.random.choice(self._num_members, size=2,
                                                        replace=False)
            diversity = self._get_member_pair_cosine_similarity(member_idx1,
                                                                member_idx2)
            loss_dict['cosine_similarity'] = diversity.item()
            total_loss += self._diversity_coef * diversity
        loss_dict['loss'] = total_loss.item()
        return total_loss, loss_dict

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
        if self._efficient_sampling:
            return self._efficient_forward(net_in, single_sample=True)
        if self._curr_sample is None:
            self._curr_sample = self._draw_from_categorical(len(net_in))
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
        if self._efficient_sampling:
            return self._efficient_forward(net_in, single_sample=False)
        if self._curr_sample is None:
            self._curr_sample = self._draw_from_categorical(len(net_in))
        elif len(self._curr_sample) < len(net_in):
            self._curr_sample = torch.cat([
                self._curr_sample,
                self._draw_from_categorical(len(net_in) - len(self._curr_sample)),
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

    def _efficient_forward(
        self,
        net_in: torch.Tensor,
        single_sample: bool,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Get the output for a single sample in the model.

        Args:
            net_in: The input for the network.
            single_sample: Whether this should come from one network.

        Returns:
            The deltas for next states and dictionary of info.
        """
        if self._curr_sample is None:
            self._curr_sample = np.arange(self._num_members)
            np.random.shuffle(self._curr_sample)
        if single_sample:
            return getattr(
                self,
                f'_member_{self._curr_sample[0]}').single_sample_output_from_torch(
                    net_in)
        input_partion = np.linspace(0, len(net_in), self._num_members+1)
        deltas = []
        info_dict = defaultdict(list)
        for part_idx, samp_idx in enumerate(self._curr_sample):
            member = getattr(self, f'_member_{samp_idx}')
            delta, info = member.multi_sample_output_from_torch(
                net_in[int(input_partion[part_idx]):int(input_partion[part_idx + 1])]
            )
            deltas.append(delta)
            for k, v in info.items():
                info_dict[k].append(v)
        return torch.cat(deltas, dim=0), {k: torch.cat(v, dim=0)
                                          for k, v in info_dict.items()}

    def _draw_from_categorical(self, num_samples) -> torch.Tensor:
        """Draw from categorical distribution.

        Args:
            num_samples: The number of samples to draw.

        Returns:
            The draws from the distribution.
        """
        return torch.randint(self._num_members, size=(num_samples,))

    @property
    def sample_mode(self) -> str:
        """The sample mode is the method that in which we get next state."""
        return self._sample_mode

    @sample_mode.setter
    def sample_mode(self, mode: str) -> None:
        """Set the sample mode to the appropriate mode."""
        self._sample_mode = mode

    @property
    def input_dim(self) -> int:
        """The sample mode is the method that in which we get next state."""
        return self._member_0.input_dim

    @property
    def output_dim(self) -> int:
        """The sample mode is the method that in which we get next state."""
        return self._member_0.output_dim

    @property
    def members(self) -> Sequence[AbstractPlModel]:
        """The members of the ensemble."""
        return [getattr(self, f'_member_{idx}')
                for idx in range(self._num_members)]

    @property
    def metrics(self) -> Dict[str, Callable[[torch.Tensor], torch.Tensor]]:
        """Get the list of metric functions to compute."""
        return self._member_0.metrics

    @property
    def learning_rate(self) -> float:
        """Get the learning rate."""
        return self._member_0.learning_rate

    @property
    def weight_decay(self) -> float:
        """Get the weight decay."""
        return self._member_0.weight_decay

    @property
    def normalizer(self) -> Normalizer:
        """Get the weight decay."""
        return self._member_0.normalizer

    def _get_member_pair_cosine_similarity(
            self,
            member_idx1: int,
            member_idx2: int,
    ) -> torch.Tensor:
        """Get cosine similarity for regularization between two members.

        Code taken from https://github.com/apple/learning-subspaces/blob/9e4cdcf4cb928
        35f8e66d5ed13dc01efae548f67/trainers/train_one_dim_subspaces.py

        Args:
            member_idx1: The index of the first ensemble member.
            member_idx2: The index of the second ensemble member.

        Result:
            The cosine similarity.
        """
        param1 = torch.nn.utils.parameters_to_vector(
                getattr(self, f'_member_{member_idx1}').parameters())
        param2 = torch.nn.utils.parameters_to_vector(
                getattr(self, f'_member_{member_idx2}').parameters())
        return self._similarity(param1, param2).pow(2)

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
        return {}
