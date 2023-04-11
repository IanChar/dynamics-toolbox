"""
Utilities for pytorch devices.
Heavily inspired by pytorch utils in rlkit.

Author: Ian Char
Date: April 6, 2023
"""
from typing import Optional

import torch
import numpy as np


class DeviceManager:

    def __init__(self):
        """Constructor"""
        self.device = 'cpu'
        self._use_gpu = False

    def set_cuda_device(self, device: Optional[int] = None):
        """Set the cuda device.

        Args:
            device: The cuda device id. If None then use cpu.
        """
        if device is None:
            self.device = 'cpu'
            self._use_gpu = False
        else:
            self.device = f'cuda:{device}'
            self._use_gpu = True

    def empty(self, *args, torch_device=None, **kwargs):
        if torch_device is None:
            torch_device = self.device
        return torch.empty(*args, **kwargs, device=torch_device)

    def from_numpy(self, *args, **kwargs):
        return torch.as_tensor(*args, **kwargs).float().to(self.device)

    def get_numpy(self, tensor: torch.Tensor):
        return tensor.to('cpu').detach().numpy()

    def randint(self, *sizes, torch_device=None, **kwargs):
        if torch_device is None:
            torch_device = self.device
        return torch.randint(*sizes, **kwargs, device=torch_device)

    def zeros(self, *sizes, torch_device=None, **kwargs):
        if torch_device is None:
            torch_device = self.device
        return torch.zeros(*sizes, **kwargs, device=torch_device)

    def ones(self, *sizes, torch_device=None, **kwargs):
        if torch_device is None:
            torch_device = self.device
        return torch.ones(*sizes, **kwargs, device=torch_device)

    def ones_like(self, *args, torch_device=None, **kwargs):
        if torch_device is None:
            torch_device = self.device
        return torch.ones_like(*args, **kwargs, device=torch_device)

    def randn(self, *args, torch_device=None, **kwargs):
        if torch_device is None:
            torch_device = self.device
        return torch.randn(*args, **kwargs, device=torch_device)

    def zeros_like(self, *args, torch_device=None, **kwargs):
        if torch_device is None:
            torch_device = self.device
        return torch.zeros_like(*args, **kwargs, device=torch_device)

    def tensor(self, *args, torch_device=None, **kwargs):
        if torch_device is None:
            torch_device = self.device
        return torch.tensor(*args, **kwargs, device=torch_device)

    def normal(self, *args, **kwargs):
        return torch.normal(*args, **kwargs).to(self.device)

    def torch_ify(self, np_array_or_other):
        if isinstance(np_array_or_other, np.ndarray):
            return self.from_numpy(np_array_or_other)
        else:
            return np_array_or_other


MANAGER = DeviceManager()
