"""
The soft actor critic algorithm.

Author: Ian Char
Date: April 6, 2023
"""
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor

from dynamics_toolbox.rl.algorithms.abstract_rl_algorithm import RLAlgorithm
from dynamics_toolbox.rl.policies.abstract_policy import Policy
from dynamics_toolbox.rl.valnet.qnet import QNet
from dynamics_toolbox.utils.pytorch.device_utils import MANAGER as dm


class SAC(RLAlgorithm):

    def __init__(
        self,
        policy: Policy,
        qnet1: QNet,
        qnet2: QNet,
        target_qnet1: QNet,
        target_qnet2: QNet,
        discount: float,
        learning_rate: float = 3e-4,
        soft_target_update_weight: float = 5e-3,
        soft_target_update_frequency: int = 1,
        target_entropy: Optional[float] = None,
        entropy_tune: bool = True,
    ):
        """Constructor.

        Args:
            policy: The policy network.
            qnet1: First Q network.
            qnet2: Second Q network.
            target_qnet1: First target Q network.
            target_qnet2: Second target Q network.
            discount: Discount factor for bellman loss.
            learning_rate: Learning rate for the optimizer.
            soft_target_update_weight: Weighting for how much to update target
                network in the soft update.
            soft_target_update_frequency: How frequently to do soft update of target
                networks.
            target_entropy: Target entropy to hit.
            entropy_tune: Whether to turn entropy tuning on.
        """
        self.policy = policy
        self.qnet1 = qnet1
        self.qnet2 = qnet2
        self.target_qnet1 = target_qnet1
        self.target_qnet2 = target_qnet2
        self.discount = discount
        self.soft_target_update_weight = soft_target_update_weight
        self.soft_target_update_frequency = soft_target_update_frequency
        params = list(policy.parameters())\
            + list(qnet1.parameters())\
            + list(qnet2.parameters())
        self._steps_since_last_soft_update = 0
        # Possibly set up entropy tuning.
        self.entropy_tune = entropy_tune
        if entropy_tune:
            if target_entropy is None:
                self.target_entropy = -policy.act_dim
            else:
                self.target_entropy = target_entropy
            self.log_alpha = dm.zeros(1, requires_grad=True)
            params.append(self.log_alpha)
        self._optimizer = torch.optim.Adam(params, lr=learning_rate)

    def _compute_losses(
        self,
        pt_batch: Dict[str, Tensor],
    ) -> Tuple[Tensor, Dict[str, float]]:
        """Compute the loasses.

        Args:
            pt_batch: Dictionary of fields w shape (batch_size, *)

        Returns: Total loss and the diictionary of loss statistics.
        """
        obs, acts, rews, nxts, terms = [pt_batch[k] for k in
                                        ('obs', 'acts', 'rews', 'nxts', 'terms')]
        loss, loss_stats = self._compute_policy_loss(obs)
        qloss, qloss_stats = self._compute_q_loss(
            obs, acts, rews, nxts, terms)
        loss_stats.update(qloss_stats)
        return loss + qloss, loss_stats

    def _compute_policy_loss(
        self,
        obs: Tensor,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """Compute the policy loss.

        Args:
            obs: The observations w shape (batch_size, obs_dim).

        Returns: The loss and the dictionary of loss stats.
        """
        pi_acts, logprobs = self.policy(obs)[:2]
        loss_stats = {}
        total_loss = 0
        if self.entropy_tune:
            mean_logprobs = logprobs.mean().item()
            total_loss -= self.log_alpha.exp() * (mean_logprobs + self.target_entropy)
            alpha = self.log_alpha.exp().item()
            loss_stats['alpha_loss'] = total_loss.item()
            loss_stats['alpha'] = alpha
        else:
            alpha = 1
        values = torch.min(self.qnet1(obs, pi_acts), self.qnet2(obs, pi_acts))
        policy_loss = (alpha * logprobs - values).mean()
        loss_stats['policy_loss'] = policy_loss.item()
        total_loss += policy_loss
        return total_loss, loss_stats

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
        q1pred = self.qnet1(obs, acts)
        q2pred = self.qnet2(obs, acts)
        nxt_acts, nxt_logpis = self.policy(nxts)[:2]
        target_qs = torch.min(
            self.target_qnet1(nxts, nxt_acts),
            self.target_qnet2(nxts, nxt_acts)
        ) - alpha * nxt_logpis
        bellman_targets = (rews + (1. - terms) * self.discount * target_qs).detach()
        q1_loss = (q1pred - bellman_targets).pow(2).mean()
        q2_loss = (q2pred - bellman_targets).pow(2).mean()
        return q1_loss + q2_loss, {
            'Q1_loss': q1_loss.item(),
            'Q2_loss': q2_loss.item(),
            'Q1_pred': q1pred.mean().item(),
            'Q2_pred': q2pred.mean().item(),
        }

    @property
    def optimizer(self):
        """Optimzier."""
        return self._optimizer

    @property
    def policy(self):
        """Optimzier."""
        return self.policy
