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
        for batch_idx, norm_pair in enumerate(norm_infos):
            self.register_buffer(f'{batch_idx}_offset', norm_pair[0].reshape(1, -1))
            self.register_buffer(f'{batch_idx}_scaling', norm_pair[1].reshape(1, -1))

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
        return ((x - getattr(self, f'{batch_idx}_offset'))
                / getattr(self, f'{batch_idx}_scaling'))

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
        return x * getattr(self, f'{batch_idx}_scaling') \
               + getattr(self, f'{batch_idx}_offset')

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
