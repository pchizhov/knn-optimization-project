import numpy as np
from abc import ABC, abstractmethod
from scipy.spatial import distance_matrix


class Optimizer(ABC):

    def __init__(self, X: np.ndarray):
        self.X = np.copy(X)

    @abstractmethod
    def query(self, X: np.ndarray, k: int) -> np.ndarray:
        pass


class LinearOptimizer(Optimizer):

    def query(self, X: np.ndarray, k=1) -> np.ndarray:
        distances = distance_matrix(X, self.X)
        indices = np.argsort(distances, axis=1)[:, :k]
        return indices
