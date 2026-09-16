"""Linear (fully connected) layer: z = xW + b."""

import numpy as np

from nn.module import Module


class Linear(Module):
    """A fully connected layer computing z = xW + b.

    Attributes:
        W (np.ndarray): weight matrix, shape (in_features, out_features).
        b (np.ndarray): bias vector, shape (out_features,).
    """

    def __init__(self, in_features:int, out_features:int) -> None:
        """Initialize the layer's weights and bias.

        Args:
            in_features (int): number of input features.
            out_features (int): number of output neurons.
        Sets:
        self.W (np.ndarray): weight matrix, shape
        (in_features, out_features). Xavier-initialized,
        not zeros (see "Weight initialization" below).
        self.b (np.ndarray): bias vector, shape
        (out_features,). Initialized to zero.
        """
        LIMIT = np.sqrt(6 / (in_features + out_features ))
        self.W = np.random.uniform(
            -LIMIT, LIMIT, size=(in_features, out_features)
        )
        self.b = np.zeros(out_features)
