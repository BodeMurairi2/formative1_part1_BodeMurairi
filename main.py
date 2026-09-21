"""Train and evaluate a single-layer linear classifier on the AND dataset."""

import numpy as np

from nn.activations.sigmoid import Sigmoid
from nn.layers.linear import Linear
from nn.losses.cross_entropy_loss import CrossEntropyLoss
from nn.optim import SGD
from nn.utils import set_seed

linear = None
sigmoid = None

def toy_data() -> tuple[np.ndarray, np.ndarray]:
    """Build the AND-gate dataset.

    Returns:
        tuple[np.ndarray, np.ndarray]: ``(x, y)``. ``x`` has shape (4, 2)
        and holds every combination of two binary inputs. ``y`` has shape
        (4, 1) and is 1.0 only when both inputs are 1.
    """
    x = np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ])
    y = np.array([
        [0.0],
        [0.0],
        [0.0],
        [1.0],
    ])
    return x, y

def train(
        epochs:int = 4000,
        lr:float=1.0,
        seed:int = 0) -> list[float]:
    """Train a single-layer linear classifier on the AND dataset.

    Builds Linear -> Sigmoid with binary cross-entropy loss and SGD, then
    runs full-batch gradient descent. The trained layers are left in the
    module-level ``linear`` and ``sigmoid`` so ``accuracy`` can use them.

    Args:
        epochs (int): number of full passes over the dataset.
        lr (float): learning rate for SGD.
        seed (int): seed for reproducible weight initialization.

    Returns:
        list[float]: the loss recorded at every epoch, in order.
    """
    global linear, sigmoid
    set_seed(seed=seed)
    x, y = toy_data()
    linear = Linear(2, 1)
    sigmoid = Sigmoid()
    cross_entropy_loss = CrossEntropyLoss()
    optimizer = SGD(parameters=linear.parameters(), lr=lr)

    history = []
    for _ in range(epochs):
        forward = linear.forward(x=x)
        predictions = sigmoid.forward(x=forward)
        loss = cross_entropy_loss.forward(predictions=predictions, targets=y)
        history.append(loss)

        loss_derivative = cross_entropy_loss.backward()
        activation_derivative = sigmoid.backward(grad_output=loss_derivative)
        linear.backward(grad_output=activation_derivative)

        optimizer.step()
        optimizer.zero_grad()

    return history


def accuracy(loss_history:list[float] = None) -> float:
    """Report the accuracy of the trained model on the AND dataset.

    Runs the model left behind by ``train`` on the toy data, thresholds the
    predicted probabilities at 0.5, and compares them with the labels.
    Call ``train`` first.

    Args:
        loss_history (list[float]): unused; kept for the required interface.

    Returns:
        float: fraction of correct predictions, between 0.0 and 1.0.
    """
    x, y = toy_data()
    z = linear.forward(x=x)
    probabilities = sigmoid.forward(x=z)
    predictions = probabilities > 0.5
    correct = predictions == (y == 1.0)
    return float(correct.sum()/correct.size)

if __name__ == "__main__":
    print("Classifier built with NN")
    model = train()
    model_accuracy = accuracy()
    print(f"Final Model Accuracy is {model_accuracy}")
    print(f"Loss: {model}")
