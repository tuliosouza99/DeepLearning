from typing import Iterable

import numpy as np
from numpy.typing import NDArray


class Perceptron:
    def __init__(self, lr=0.01, n_iter=50, seed=42):
        self.lr = lr
        self.n_iter = n_iter
        self.seed = seed

    def train(self, X: NDArray, y: NDArray):
        rng = np.random.default_rng(self.seed)
        self.w_ = rng.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = np.float32(0.0)
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.lr * (target - self.predict(xi))
                self.w_ += update * xi
                self.b_ += update
                errors += int(update != 0.0)
            self.errors_.append(errors)

        return self

    def net_input(self, X: NDArray):
        """Calculate net input"""
        return np.dot(X, self.w_) + self.b_

    def predict(self, X: NDArray):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)


def multiclass_train(
    X: NDArray, y: NDArray, n_iter: int = 50, lr: float = 0.01, seed: int = 42
):
    n_classes = len(np.unique(y))
    perceptrons = [
        Perceptron(lr=lr, n_iter=n_iter, seed=seed) for _ in range(n_classes)
    ]

    for i, perceptron in enumerate(perceptrons):
        perceptron.train(X, np.where(y == i, 1, -1))

    return perceptrons


def multiclass_predict(X: NDArray, perceptrons: Iterable[Perceptron]):
    return [
        np.argmax([perceptron.predict(xi) for perceptron in perceptrons]) for xi in X
    ]
