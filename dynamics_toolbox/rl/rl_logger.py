"""
Logger for training Reinforcement Learning.

Author: Ian Char
Date: April 7, 2023
"""
from typing import Dict, Optional
import os

import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from dynamics_toolbox.rl.modules.policies.abstract_policy import Policy


class RLLogger:

    def __init__(
        self,
        run_dir: str,
        stats_file: Optional[str] = 'stats.txt',
        checkpoint_policy_every: Optional[int] = None,
        record_tensorboard: bool = True,
        show_pbar: bool = True,
    ):
        """Constructor.

        Args:
            run_dir: The directory to log to.
            stats_file: The file to record statistics to. If None then do not
                record to stats_file.
            checkpoint_policy_every: How often to checkpoint policy. If None, do not
                checkpoint the policy until the end.
            record_tensorboard: Whether to record tensorboard.
        """
        self.run_dir = run_dir
        self.stats_file = stats_file
        self.checkpoint_policy_every = checkpoint_policy_every
        self.show_pbar = show_pbar
        if record_tensorboard:
            self._summary_writer = SummaryWriter(run_dir)
        else:
            self._summary_writer = None
        self.headers = None
        self.pbar = None
        self.pbar_inner = None
        self.num_inner_loops = 0
        self._last_returns = (0.0, 0.0)
        if checkpoint_policy_every is not None:
            os.mkdir(os.path.join(run_dir, 'checkpoints'))

    def start(self, num_epochs: int):
        """Start logging.

        Args:
            num_epochs: The number of epochs expected.
        """
        if self.show_pbar:
            self.pbar = tqdm(total=num_epochs, position=0)

    def log_epoch(
        self,
        epoch: int,
        num_steps: int,
        stats: Dict[str, float],
        returns_mean: Optional[float] = None,
        returns_std: Optional[float] = None,
        policy: Optional[Policy] = None,
    ):
        """Log the current epoch.

        Args:
            epoch: The current epoch.
            num_steps: The current number of steps.
            stats: The stats for this epoch.
            returns_mean: The policy mean.
            returns_std: The policy std.
            policy: The policy.
        """
        if returns_mean is not None:
            self._last_returns = (returns_mean, returns_std)
        # If first run then set up.
        if self.headers is None:
            self.headers = ['Epoch', 'Samples', 'Returns/Mean', 'Returns/Std']
            self.headers += list(stats.keys())
            if self.stats_file is not None:
                with open(os.path.join(self.run_dir, self.stats_file), 'w') as sfile:
                    sfile.write(','.join(self.headers + ['\n']))
        # Write to stats file.
        if self.stats_file is not None:
            data_str = ','.join([
                str(epoch), str(num_steps),
                f'{returns_mean:0.2f}' if returns_mean is not None else '',
                f'{returns_std:0.2f}' if returns_std is not None else ''
            ] + [f'{stats[k]:0.2f}' for k in self.headers[4:]] + ['\n'])
            with open(os.path.join(self.run_dir, self.stats_file), 'a') as sfile:
                sfile.write(data_str)
        # Write to tensorboard.
        if self._summary_writer is not None:
            if returns_mean is not None:
                self._summary_writer.add_scalar('Eval/returns_mean', returns_mean,
                                                num_steps)
            if returns_std is not None:
                self._summary_writer.add_scalar('Eval/returns_std', returns_std,
                                                num_steps)
            for k, v in stats.items():
                self._summary_writer.add_scalar(k, v, num_steps)
        # Possibly checkpoint policy.
        if (self.checkpoint_policy_every is not None
                and (epoch % self.checkpoint_policy_every) == 0):
            torch.save(policy.state_dict(), f'checkpoints/epoch_{epoch}.pt')
        # Update pbar.
        if self.pbar is not None:
            self.pbar.update(1)
            self.pbar.set_postfix_str(f'Returns: {self._last_returns[0]:0.2f} '
                                      f'+- {self._last_returns[1]:0.2f}')

    def set_phase(
        self,
        phase: str,
    ):
        """Set the phase. Right now just reflected in pbar.

        Args:
            status: What is currently happening.
        """
        if self.pbar_inner is not None:
            self.pbar_inner.set_description(phase)
        elif self.pbar is not None:
            self.pbar.set_description(phase)

    def start_inner_loop(
        self,
        desc: str,
        num_loops: int,
    ):
        """Start inner progress bar.

        Args:
            desc: Description.
            num_loops: Number of inner loops.
        """
        if self.show_pbar:
            self.pbar_inner = tqdm(total=num_loops, position=1, leave=False)
            self.num_inner_loops = num_loops

    def end_inner_loop(self):
        """Mark the end of an inner loop"""
        if self.pbar_inner is not None:
            self.pbar_inner.update(1)
            self.num_inner_loops -= 1
            if self.num_inner_loops == 0:
                self.pbar_inner.close()
                self.pbar_inner = None

    def end(self, policy: Policy):
        """Called at the end of the run.

        Args:
            policy: The final policy.
        """
        torch.save(policy.state_dict(), 'policy.pt')
        if self.pbar is not None:
            self.pbar.close()
            self.pbar = None
        if self.pbar_inner is not None:
            self.pbar_inner.close()
