"""
Classes for normalizing data.

Author: Ian Char
"""
from typing import Sequence

import torch

class Normalizer(torch.nn.Module):

    def __init__(
            self,
            x_offset: torch.Tensor,
            x_scaling: torch.Tensor,
            y_offset: torch.Tensor,
            y_scaling: torch.Tensor,
    ):
        """Constructor.

        Args:
            x_offset: The x offset to apply.
            x_offset: The x scaling to apply.
            y_offset: The y offset to apply.
            y_offset: The y scaling to apply.
        """
        super().__init__()
        self.register_buffer('x_offset', x_offset.reshape(1, -1))
        self.register_buffer('x_scaling', x_scaling.reshape(1, -1))
        self.register_buffer('y_offset', y_offset.reshape(1, -1))
        self.register_buffer('y_scaling', y_scaling.reshape(1, -1))

    def transform_batch(self, batch: Sequence[torch.Tensor]) ->\
            Sequence[torch.Tensor]:
        """Transform a batch into normalized space.

        Args:
            The batch as input and output data.

        Returns:
            The transformed batch.
        """
        new_input = (batch[0] - self.x_offset) / self.x_scaling
        new_output = (batch[1] - self.y_offset) / self.y_scaling
        return [new_input, new_output] + batch[2:]

    def untransform_output(self, output: torch.Tensor) -> torch.Tensor:
        """Untransform the output.

        Args:
            output: The output to transform.

        Returns:
            The transformed output.
        """
        return output * self.y_scaling + self.y_offset

