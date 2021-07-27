"""
Early stopping version that monitors multiples quantities.

Author: Ian Char
"""
from typing import List, Dict, Any, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping


class MultiMonitorEarlyStopping(EarlyStopping):

    def __init__(
            self,
            monitors: List[str],
            multi_mode: str = 'all',
            **kwargs
    ):
        """
        Constructor.
        Args:
            monitors: List of quantities to monitor.
            multi_mode: Whether to wait until 'all' conditions met or 'any'.
            **kwargs:
        """
        super().__init__(monitor=monitors[0], **kwargs)
        self._monitors = monitors
        if multi_mode != 'all' and multi_mode != 'any':
            raise ValueError(f'multi_mode must either be all or any. Got {multi_mode}.')
        self._multi_mode = multi_mode
        self._should_stops = [False for _ in self._monitors]
        torch_inf = torch.tensor(np.Inf)
        self.best_scores = [torch_inf if self.monitor_op == torch.lt else -torch_inf
                            for _ in self._monitors]

    def on_save_checkpoint(
            self,
            trainer: 'pl.Trainer',
            pl_module: 'pl.LightningModule',
            checkpoint: Dict[str, Any],
    ) -> Dict[str, Any]:
        to_return = {
            'wait_count': self.wait_count,
            'stopped_epoch': self.stopped_epoch,
            'patience': self.patience
        }
        for monitor_idx, best in enumerate(self.best_scores):
            to_return[f'best_score_{monitor_idx}'] = best
        return to_return

    def on_load_checkpoint(self, callback_state: Dict[str, Any]) -> None:
        self.wait_count = callback_state['wait_count']
        self.stopped_epoch = callback_state['stopped_epoch']
        self.patience = callback_state['patience']
        self.best_scores = [callback_state[f'best_score_{monitor_idx}']
                            for monitor_idx in range(len(self._monitors))]

    def _run_early_stopping_check(self, trainer: 'pl.Trainer') -> None:
        """
        Checks whether the early stopping condition is met
        and if so tells the trainer to stop the training.
        """
        logs = trainer.callback_metrics

        if (
                trainer.fast_dev_run  # disable early_stopping with fast_dev_run
                or not self._validate_condition_metric(logs)  # short circuit if metric not present
        ):
            return
        reason = ...
        for monitor_idx, monitor in enumerate(self._monitors):
            current = logs.get(monitor)
            # when in dev debugging
            trainer.dev_debugger.track_early_stopping_history(self, current)
            should_stop, reason = self._evalute_stopping_criteria(current, monitor_idx)
            self._should_stops[monitor_idx] = self._should_stops[monitor_idx] or should_stop
        if self._multi_mode == 'all':
            should_stop = np.all(self._should_stops)
        else:
            should_stop = np.any(self._should_stops)
        # stop every ddp process if any world process decides to stop
        should_stop = trainer.training_type_plugin.reduce_boolean_decision(should_stop)
        trainer.should_stop = trainer.should_stop or should_stop
        if should_stop:
            self.stopped_epoch = trainer.current_epoch
        if reason and self.verbose:
            self._log_info(trainer, reason)

    def _evalute_stopping_criteria(self, current: torch.Tensor, monitor_idx: int) -> Tuple[bool, str]:
        should_stop = False
        reason = None
        monitor = self._monitors[monitor_idx]
        best_score = self.best_scores[monitor_idx]
        if self.check_finite and not torch.isfinite(current):
            should_stop = True
            reason = (
                f"Monitored metric {monitor} = {current} is not finite."
                f" Previous best value was {best_score:.3f}. Signaling Trainer to stop."
            )
        elif self.stopping_threshold is not None and self.monitor_op(current, self.stopping_threshold):
            should_stop = True
            reason = (
                "Stopping threshold reached:"
                f" {monitor} = {current} {self.order_dict[self.mode]} {self.stopping_threshold}."
                " Signaling Trainer to stop."
            )
        elif self.divergence_threshold is not None and self.monitor_op(-current, -self.divergence_threshold):
            should_stop = True
            reason = (
                "Divergence threshold reached:"
                f" {monitor} = {current} {self.order_dict[self.mode]} {self.divergence_threshold}."
                " Signaling Trainer to stop."
            )
        elif self.monitor_op(current - self.min_delta, self.best_score.to(current.device)):
            should_stop = False
            reason = self._improvement_message(current)
            self.best_scores[monitor_idx] = current
            self.wait_count = 0
        else:
            self.wait_count += 1
            if self.wait_count >= self.patience:
                should_stop = True
                reason = (
                    f"Monitored metric {monitor} did not improve in the last {self.wait_count} records."
                    f" Best score: {best_score:.3f}. Signaling Trainer to stop."
                )

        return should_stop, reason
