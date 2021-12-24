import numpy as np
from typing import Type
import scipy.stats as ss
from optimizer import Optimizer


class KNNClassifier:

    def __init__(self, k: int, optimizer: Type[Optimizer]):
        self.k = k
        self.optimizer = optimizer(k)

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y = y
        self.optimizer.fit(X, y)

    def predict(self, X: np.ndarray) -> int:
        nearest = self.optimizer.get_k_nearest(X)
        return ss.mode(nearest).mode
