"""Binary cross-entropy loss."""

import numpy as np


class CrossEntropyLoss:
    """Binary cross-entropy loss for a single output probability.

    Does not subclass Module -- see the note below.
    """
    def forward(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Compute the average binary cross-entropy loss.

        Args:
        predictions (np.ndarray): predicted probabilities,
        shape (m,) or (m, 1). Clip away from exactly
        0 or 1 before use -- see "Numerical stability"
        above.
        targets (np.ndarray): true labels, same shape as
        predictions, values 0 or 1.

        Returns:
        float: the scalar loss, averaged over the batch.
        """
        eps = 1e-12
        predictions = np.clip(predictions, eps, 1 - eps)
        self.predictions = predictions
        self.targets = targets
        self.nbr_rows = targets.shape[0]
        positive_term = targets * np.log(predictions)
        negative_term = (1 - targets) * np.log(1 - predictions)
        return float((-1 / self.nbr_rows) * np.sum(positive_term + negative_term))

    def backward(self) -> np.ndarray:
        """Compute the gradient of the loss w.r.t. predictions.

        Returns:
        np.ndarray: dL/da, same shape as the predictions
        passed to forward. Use the same clipped
        predictions here as in forward -- see
        "Numerical stability" above.
        """
        grad_positive = self.targets/self.predictions
        grad_negative = (1 - self.targets)/(1 - self.predictions)
        return (-1/self.nbr_rows)*(grad_positive - grad_negative)
