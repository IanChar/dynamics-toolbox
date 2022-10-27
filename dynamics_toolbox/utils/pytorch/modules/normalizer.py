"""
Classes for normalizing data.

Author: Ian Char
"""
from typing import Sequence, Tuple

import torch


class Normalizer(torch.nn.Module):

    def __init__(
            self,
            norm_infos: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    ):
        """Constructor.

        Args:
            norm_infos: Sequence of tuples of (offset, scaling) for each item
                that appears in a batch for learning.
        """
        super().__init__()
        self._num_normalizers = len(norm_infos)
        for batch_idx, norm_pair in enumerate(norm_infos):
            offsets = norm_pair[0].reshape(1, -1)
            scales = norm_pair[1].reshape(1, -1)
            for sidx, scale in enumerate(norm_pair[1].flatten()):
                if scale < 1e-8:
                    scales[0, sidx] = 1
            self.register_buffer(f'{batch_idx}_offset', offsets)
            self.register_buffer(f'{batch_idx}_scaling', scales)

    def normalize_batch(self, batch: Sequence[torch.Tensor]) -> \
            Sequence[torch.Tensor]:
        """Transform a batch into normalized space.

        Args:
            The batch as input and output data.

        Returns:
            The transformed batch.
        """
        return [self.normalize(elem, bidx) for bidx, elem in enumerate(batch)]

    def normalize(
            self,
            x: torch.Tensor,
            batch_idx: int,
    ) -> torch.Tensor:
        """Normalize an element in the batch.

        Args:
            x: The input to normalize.
            batch_idx: The position this quantity appears in for the batch.

        Returns:
            The transformed input.
        """
        if batch_idx >= self._num_normalizers:
            return x
        orig_shape = x.shape
        if len(x.shape) > 2:
            x = x.reshape(-1, orig_shape[-1])
        return ((x - getattr(self, f'{batch_idx}_offset'))
                / getattr(self, f'{batch_idx}_scaling')).reshape(orig_shape)

    def unnormalize(
            self,
            x: torch.Tensor,
            batch_idx: int,
    ) -> torch.Tensor:
        """Unnormalize an element in the batch.

        Args:
            x: The input to unnormalize.
            batch_idx: The position this quantity appears in for the batch.

        Returns:
            The transformed input.
        """
        if batch_idx >= self._num_normalizers:
            return x
        orig_shape = x.shape
        if len(x.shape) > 2:
            x = x.reshape(-1, orig_shape[-1])
        return (x * getattr(self, f'{batch_idx}_scaling')
                + getattr(self, f'{batch_idx}_offset')).reshape(orig_shape)


class NoNormalizer(Normalizer):

    def __init__(self):
        """Constructor."""
        super().__init__([])

    def normalize(
            self,
            x: torch.Tensor,
            batch_idx: int,
    ) -> torch.Tensor:
        """Normalize an element in the batch.

        Args:
            x: The input to normalize.
            batch_idx: The position this quantity appears in for the batch.

        Returns:
            The transformed input.
        """
        return x

    def unnormalize(
            self,
            x: torch.Tensor,
            batch_idx: int,
    ) -> torch.Tensor:
        """Unnormalize an element in the batch.

        Args:
            x: The input to unnormalize.
            batch_idx: The position this quantity appears in for the batch.

        Returns:
            The transformed input.
        """
        return x
