"""
A random probability model for testing. Proxy for a tearing mode model.

Author: Rohit Sonker
"""

from typing import Tuple, Dict, Any
import torch
import numpy 

class TearingProbabilityRandomModel():
    """A random probability model for testing. Proxy for a tearing mode model."""

    def __init__(self):
        """Initialize the model."""
        # pass
        self.random_prob = 0.1

    def single_sample_output_from_torch(
            self,
            # net_in: torch.Tensor
    ) -> numpy.ndarray:
        """Get the output for a single sample in the model.

        Args:
            net_in: The input for the network.

        Returns:
            The predictions for a single function sample
        """
        # with torch.no_grad():
        #     predictions = torch.rand((net_in.shape[0], 2))
        #     pred_class = torch.argmax(predictions, dim=1).numpy()
        # return the prob with mean around self.random_prob

        predictions = numpy.random.normal(self.random_prob, 0.1, 1)
        return predictions

    def multi_sample_output_from_torch(
            self,
            net_in: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Get the output where each input is assumed to be from a different sample.

        Args:
            net_in: The input for the network.

        Returns:
            The deltas for next states and dictionary of info.
        """
        return self.single_sample_output_from_torch(net_in)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function for network.

        Args:
            x: The input to the network.

        Returns:
            The output of the network.
        """
        return self.single_sample_output_from_torch(x)[0]
    
    
    def get_tearing_probability(self, x):
        betan_prob = numpy.array([0.1 for _ in range(len(x))])
        # betan_prob[numpy.where(x>3)] = 0.5
        betan_prob[numpy.where(x>3.5)] = 0.4
        betan_prob[numpy.where(x>4)] = 0.9

        prob = numpy.random.normal(loc = betan_prob, scale = 0.1)

        return numpy.clip(prob, 0, 1)