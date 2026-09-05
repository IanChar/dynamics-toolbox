"""
Physics-informed variants of the sequential dynamics models.

The mixin adds config-driven soft physics penalties on top of the base
model's data loss. The physics term itself arrives via a hydra target
(``physics_loss_cfg``), so this module stays domain-agnostic — e.g.
FusionControl passes fusion_control.dynamics.physics.mhd_losses.MHDPhysicsLoss.

The physics module must be a torch.nn.Module with signature
    forward(mean, batch, normalizer) -> (loss_tensor, stats_dict)
operating on the normalized batch/prediction (it is handed the model's
normalizer to invert). It is registered as a submodule so its buffers follow
the model between devices. Physics acts on the predicted mean only — use it
for Stage-1 (MSE) training; leave it off for logvar-only Stage 2.
"""
from typing import Dict, Optional, Sequence, Tuple

import hydra.utils
import torch
from omegaconf import DictConfig

from dynamics_toolbox.models.pl_models.sequential_models.rpnn import RPNN
from dynamics_toolbox.models.pl_models.sequential_models.tpnn import TPNN


class PhysicsInformedMixin:
    """Adds a weighted physics penalty to the base model's loss()."""

    def __init__(self, *args, physics_loss_cfg: Optional[DictConfig] = None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        if physics_loss_cfg is not None:
            self.physics_loss = hydra.utils.instantiate(
                physics_loss_cfg, _recursive_=False)
        else:
            self.physics_loss = None

    def loss(self, net_out: Dict[str, torch.Tensor],
             batch: Sequence[torch.Tensor]) -> \
            Tuple[torch.Tensor, Dict[str, float]]:
        loss, stats = super().loss(net_out, batch)
        if self.physics_loss is not None:
            p_loss, p_stats = self.physics_loss(
                net_out["mean"], batch, self.normalizer)
            loss = loss + p_loss
            stats.update(p_stats)
            stats["loss"] = loss.item()
        return loss, stats


class PIRPNN(PhysicsInformedMixin, RPNN):
    """RPNN + physics penalties."""


class PITPNN(PhysicsInformedMixin, TPNN):
    """TPNN + physics penalties."""
