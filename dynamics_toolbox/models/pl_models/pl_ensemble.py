"""
Ensemble of other pytorch models.

Author: Ian Char
"""
import argparse
from typing import List, Optional, Sequence, NoReturn, Dict

import numpy as np
import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.argparse import from_argparse_args

from dynamics_toolbox.constants import sampling_modes
from dynamics_toolbox.models.ensemble import FiniteEnsemble
from dynamics_toolbox.models.pl_models import PL_MODELS
from dynamics_toolbox.models.pl_models.abstract_pl_model import AbstractPlModel


class FinitePlEnsemble(LightningModule, FiniteEnsemble):

    def __init__(
            self,
            member_config: argparse.Namespace,
            members: Optional[List[AbstractPlModel]] = None,
            sample_mode: str = sampling_modes.SAMPLE_MEMBER_EVERY_TRAJECTORY,
            elite_idxs: Optional[List[int]] = None,
            num_elites: Optional[int] = None,
    ):
        """Constructor
        Args:
            member_config: The hyperparameters for a member of the ensemble
                plus the number of ensemble members.
            members: The members of the ensemble
            sample_mode: The method to use for sampling.
            elite_idxs: The indexes of the elite members. These are the members to sample from.
            num_elites: The number of elites to keep track of.
        """
        LightningModule.__init__(self)
        if members is None:
            members = [from_argparse_args(
                PL_MODELS[member_config.model_type],
                member_config)
                for _ in range(member_config.num_ensemble_members)]
        for member_idx, member in enumerate(members):
            setattr(self, f'member_{member_idx}', member)
        self._num_members = len(members)
        self._sample_mode = sample_mode
        self._nxt_member_to_sample = np.random.randint(self._num_members)
        self._num_elites = self._num_members if num_elites is None else num_elites
        if elite_idxs is None:
            self._elite_idxs = list(range(self._num_elites))
        else:
            self._elite_idxs = elite_idxs

    def training_step(
            self,
            batch: Sequence[torch.Tensor],
            batch_idx: int,
            optimizer_idx: int
    ) -> torch.Tensor:
        """Training step for pytorch lightning. Returns the loss."""
        net_out = self.members[optimizer_idx].get_net_out(batch)
        loss, loss_dict = self.members[optimizer_idx].loss(net_out, batch)
        self._log_stats(loss_dict, prefix=f'member{optimizer_idx}/train')
        return loss

    def validation_step(
            self,
            batch: Sequence[torch.Tensor],
            batch_idx: int,
    ) -> NoReturn:
        """Validation step for pytorch lightning."""
        losses = []
        for member_idx, member in enumerate(self.members):
            net_out = member.get_net_out(batch)
            loss, loss_dict = member.loss(net_out, batch)
            self._log_stats(loss_dict, prefix=f'member{member_idx}/val')
            losses.append(loss.item())
        self._elite_idxs = np.argsort(losses)[:self._num_elites]

    def test_step(
            self,
            batch: Sequence[torch.Tensor],
            batch_idx: int,
    ) -> NoReturn:
        """Validation step for pytorch lightning."""
        for member_idx, member in enumerate(self.members):
            net_out = member.get_net_out(batch)
            loss, loss_dict = member.loss(net_out, batch)
            self._log_stats(loss_dict, prefix=f'member{member_idx}/test')

    def configure_optimizers(self) -> List[torch.optim.Optimizer]:
        """
        Get the optimizers.
        Returns:
            List of the optimizers of the members.
        """
        return [member.configure_optimizers() for member in self.members]

    def _log_stats(self, *args: Dict[str, float], prefix='train', **kwargs) -> NoReturn:
        """Log all of the stats from dictionaries.
        Args:
            args: Dictionaries of torch tensors to add stats about.
            prefix: The prefix to add to the statistic string.
            kwargs: Other kwargs to be passed to self.log.
        """
        for arg in args:
            for stat_name, stat in arg.items():
                self.log(f'{prefix}/{stat_name}', stat, **kwargs)

    @property
    def members(self) -> Sequence[AbstractPlModel]:
        """The members of the ensemble."""
        return [getattr(self, f'member_{idx}')
                for idx in range(self._num_members)]

