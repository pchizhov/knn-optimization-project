import numpy as np
from abc import ABC, abstractmethod


class Optimizer(ABC):

    def __init__(self, X: np.ndarray):
        self.X = np.copy(X)

    @abstractmethod
    def query(self, X: np.ndarray, k: int) -> np.ndarray:
        pass
