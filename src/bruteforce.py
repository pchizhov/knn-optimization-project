import numpy as np
from scipy.spatial import distance_matrix
from .optimizer import Optimizer


class LinearOptimizer(Optimizer):

    def query(self, X: np.ndarray, k=1) -> np.ndarray:
        distances = distance_matrix(X, self.X)
        indices = np.argpartition(distances, k-1, axis=1)[:, :k]
        return indices

