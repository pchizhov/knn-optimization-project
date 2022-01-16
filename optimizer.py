import numpy as np
from abc import ABC, abstractmethod


class Optimizer(ABC):

    def __init__(self, k: int):
        self.k = k

    def fit(self, X: np.ndarray):
        self.X = X

    @abstractmethod
    def get_k_nearest(self, X: np.ndarray) -> np.ndarray:
        pass


class LinearOptimizer(Optimizer):

    def get_k_nearest(self, X: np.ndarray) -> np.ndarray:
        distances = np.linalg.norm(self.X - X, axis=1)
        indices = np.argsort(distances)[:self.k]
        return indices
