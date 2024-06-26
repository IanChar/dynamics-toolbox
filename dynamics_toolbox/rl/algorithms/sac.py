"""
The soft actor critic algorithm.

Author: Ian Char
Date: April 6, 2023
"""
from copy import deepcopy
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from dynamics_toolbox.rl.algorithms.abstract_rl_algorithm import RLAlgorithm
from dynamics_toolbox.rl.modules.policies import Policy
from dynamics_toolbox.rl.modules.valnets import QNet
from dynamics_toolbox.rl.util.misc import soft_update_net
from dynamics_toolbox.utils.pytorch.device_utils import MANAGER as dm
from dynamics_toolbox.utils.pytorch.weight_inits import init_net


class SAC(RLAlgorithm):

    def __init__(
        self,
        policy: Policy,
        qnet: QNet,
        discount: float,
        policy_learning_rate: float = 3e-4,
        q_learning_rate: float = 3e-4,
        soft_target_update_weight: float = 5e-3,
        soft_target_update_frequency: int = 1,
        target_entropy: Optional[float] = None,
        entropy_tune: bool = True,
        num_qnets: int = 2,
        **kwargs
    ):
        """Constructor.

        Args:
            policy: The policy network.
            qnet: The Q network. This will be copied into N q networks along with
                corresponding target networks.
            discount: Discount factor for bellman loss.
            policy_learning_rate: Learning rate for the policy optimizer.
            q_learning_rate: Learning rate for the q optimizer.
            soft_target_update_weight: Weighting for how much to update target
                network in the soft update.
            soft_target_update_frequency: How frequently to do soft update of target
                networks.
            target_entropy: Target entropy to hit.
            entropy_tune: Whether to turn entropy tuning on.
            num_qnets: Number of q networks to use. The more qnetworks usually the more
                robust the algorithm will be to q explosion.
        """
        self._policy = policy
        self.policy.to(dm.device)
        assert num_qnets >= 1, 'Requires at least 1 qnetwork.'
        self.num_qnets = num_qnets
        qnets, target_qnets = [], []
        for nq in range(num_qnets):
            newnet, new_target = [deepcopy(qnet) for _ in range(2)]
            newnet.to(dm.device)
            new_target.to(dm.device)
            newnet.apply(init_net)
            new_target.apply(init_net)
            qnets.append(newnet)
            target_qnets.append(new_target)
        self.qnets = nn.ModuleList(qnets)
        self.target_qnets = target_qnets
        self.discount = discount
        self.soft_target_update_weight = soft_target_update_weight
        self.soft_target_update_frequency = soft_target_update_frequency
        self._steps_since_last_soft_update = 0
        # Set up optimizers.
        self._policy_optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=policy_learning_rate,
        )
        self._q_optimizer = torch.optim.Adam(
            self.qnets.parameters(),
            lr=q_learning_rate,
        )
        # Possibly set up entropy tuning.
        self.entropy_tune = entropy_tune
        if entropy_tune:
            if target_entropy is None:
                self.target_entropy = -policy.act_dim
            else:
                self.target_entropy = target_entropy
            self.log_alpha = dm.zeros(1, requires_grad=True)
            self._alpha_optimizer = torch.optim.Adam(
                [self.log_alpha],
                lr=policy_learning_rate)
        else:
            self._alpha_optimizer = None

    def _compute_losses(
        self,
        pt_batch: Dict[str, Tensor],
    ) -> Dict[str, float]:
        """Compute the loasses.

        Args:
            pt_batch: Dictionary of fields w shape (batch_size, *)

        Returns: Dictionary of loss statistics.
        """
        obs, acts, rews, nxts, terms = [pt_batch[k] for k in
                                        ('observations', 'actions', 'rewards',
                                         'next_observations', 'terminals')]
        stats = {}
        # Policy loss.
        loss, loss_stats = self._compute_policy_loss(obs)
        stats.update(loss_stats)
        self._policy_optimizer.zero_grad()
        loss.backward()
        self._policy_optimizer.step()
        # QNet loss.
        loss, loss_stats = self._compute_q_loss(obs, acts, rews, nxts, terms)
        stats.update(loss_stats)
        self._q_optimizer.zero_grad()
        loss.backward()
        self._q_optimizer.step()
        # Alpha loss.
        if self.entropy_tune:
            loss, loss_stats = self._compute_alpha_loss(
                stats['Policy/logprob_mean'])
            stats.update(loss_stats)
            self._alpha_optimizer.zero_grad()
            loss.backward()
            self._alpha_optimizer.step()
        # Soft update to target networks.
        self._steps_since_last_soft_update += 1
        if self._steps_since_last_soft_update % self.soft_target_update_frequency == 0:
            self._steps_since_last_soft_update = 0
            for qnet, qtarget in zip(self.qnets, self.target_qnets):
                soft_update_net(qtarget, qnet, self.soft_target_update_weight)
        return stats

    def _compute_alpha_loss(
        self,
        mean_logprobs: float,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """Compute the policy loss.

        Args:
            obs: The observations w shape (batch_size, obs_dim).
            mean_logprobs: The current mean log probabilities for the policy actions.

        Returns: The loss and the dictionary of loss stats.
        """
        assert self.entropy_tune, 'Should not have been called if not tuning.'
        loss_stats = {}
        loss = -self.log_alpha.exp() * (mean_logprobs + self.target_entropy)
        loss_stats['Alpha/alpha_loss'] = loss.item()
        loss_stats['Alpha/alpha'] = self.log_alpha.exp().item()
        return loss, loss_stats

    def _compute_policy_loss(
        self,
        obs: Tensor,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """Compute the policy loss.

        Args:
            obs: The observations w shape (batch_size, obs_dim).

        Returns: The loss and the dictionary of loss stats.
        """
        loss_stats = {}
        pi_acts, logprobs = self.policy(obs)[:2]
        if self.entropy_tune:
            alpha = self.log_alpha.exp().item()
        else:
            alpha = 1
        values = torch.min(torch.stack([
            qnet(obs, pi_acts) for qnet in self.qnets
        ]), dim=0)[0]
        loss = (alpha * logprobs - values).mean()
        loss_stats['Policy/policy_loss'] = loss.item()
        loss_stats['Policy/logprob_mean'] = logprobs.mean().item()
        return loss, loss_stats

    def _compute_q_loss(
        self,
        obs: Tensor,
        acts: Tensor,
        rews: Tensor,
        nxts: Tensor,
        terms: Tensor,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """Compute the Q Loss

        Args:
            obs: The observations w shape (batch_size, obs_dim).
            acts: The actions w shape (batch_size, act_dim).
            rews: The rewards w shape (batch_size, 1).
            nxts: The next observations w shape (batch_size, obs_dim).
            terms: The terminals w shape (batch_size, 1).

        Returns: The loss and the dictionary of loss stats.
        """
        alpha = self.log_alpha.exp().item() if self.entropy_tune else 1
        qpreds = [qnet(obs, acts) for qnet in self.qnets]
        nxt_acts, nxt_logprobs = self.policy(nxts)[:2]
        qtarget_preds = [tqnet(nxts, nxt_acts) for tqnet in self.target_qnets]
        target_qs = torch.min(torch.stack(qtarget_preds), dim=0)[0]
        target_qs -= alpha * nxt_logprobs
        bellman_targets = (rews + (1. - terms) * self.discount * target_qs).detach()
        losses = [(qpred - bellman_targets).pow(2).mean()
                  for qpred in qpreds]
        loss_dict = {}
        qidx = 1
        for qpred, qloss, qtarget in zip(qpreds, losses, qtarget_preds):
            loss_dict.update({
                f'Q/Q{qidx}_loss': qloss.item(),
                f'Q/Q{qidx}_pred': qpred.mean().item(),
                f'Q/Q{qidx}_target_pred': qtarget.mean().item(),
            })
            qidx += 1
        return torch.sum(torch.stack(losses), dim=0), loss_dict

    @property
    def policy(self):
        """Optimzier."""
        return self._policy
