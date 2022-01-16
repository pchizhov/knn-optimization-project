import numpy as np
import scipy.stats as ss
from optimizer import Optimizer


class KNNClassifier:

    def __init__(self, k: int, optimizer: Optimizer):
        self.k = k
        self.optimizer = optimizer

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y = y
        self.optimizer.fit(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        predictions = np.zeros(X.shape[0])
        for i in range(len(X)):
            indices = self.optimizer.get_k_nearest(X[i])
            predictions[i] = ss.mode(self.y[indices]).mode
        return predictions
