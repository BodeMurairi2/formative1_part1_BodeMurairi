"""Categorical cross-entropy loss, for one-hot multi-class targets."""

import numpy as np


class CategoricalCrossEntropyLoss:
    """Categorical cross-entropy loss over C classes.

    Does not subclass Module -- see the note in Chapter 6.
    """
    def forward(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Compute the average categorical cross-entropy loss.

        Args:
        predictions (np.ndarray): softmax probabilities,
        shape (m, C). Clip away from exactly 0 before
        use -- see "The same clipping requirement as
        Chapter 6" above.
        targets (np.ndarray): one-hot true labels, shape
        (m, C).

        Returns:
        float: the scalar loss, averaged over the batch.
        """
        eps = 1e-12
        predictions = np.clip(predictions, eps, 1 - eps)
        self.predictions = predictions
        self.targets = targets
        self.nbr_rows = targets.shape[0]
        log_likelihood = np.sum(self.targets * np.log(self.predictions))
        return float((-1 / self.nbr_rows) * log_likelihood)

    def backward(self) -> np.ndarray:
        """Compute the gradient of the loss w.r.t. predictions.

        Returns:
        np.ndarray: dL/da, shape (m, C), same shape as the
        predictions passed to forward. Use the same
        clipped predictions here as in forward.
        """
        return (-1/self.nbr_rows)*(self.targets/self.predictions)
