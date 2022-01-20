import numpy as np
import scipy.stats as ss
from .optimizer import Optimizer


class KNNClassifier:

    def fit(self, X: np.ndarray, y: np.ndarray, optimizer: Optimizer):
        self.X = X
        self.y = y
        self.optimizer = optimizer

    def predict(self, X: np.ndarray, k: int) -> np.ndarray:
        predictions = np.zeros(X.shape[0])
        indices = self.optimizer.query(X, k)
        for i, idx in enumerate(indices):
            predictions[i] = ss.mode(self.y[idx]).mode
        return predictions
