import os
import re
import glob
import operator
from PIL.Image import Image
from optimizer import Optimizer
from typing import Any
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
import heapq


@dataclass
class KDTreeNode:
    is_leaf: bool
    level: int
    points: np.ndarray
    indices: np.ndarray
    left: Any
    right: Any
    parent: Any = None
    visited: bool = False

    def __post_init__(self):
        if self.left is not None:
            self.left.parent = self
        if self.right is not None:
            self.right.parent = self


class KDTreeOptimizer(Optimizer):

    def __init__(self, X: np.ndarray, leaf_size: int = 1):
        super().__init__(X)
        self.depth = 0
        self.leaf_size = leaf_size
        X_ = np.hstack((X, np.arange(len(X)).reshape(-1, 1)))
        self.root = self.kd_tree(X_)

    def kd_tree(self, X: np.ndarray, level: int = 0):
        if not len(X):
            return
        if len(X) <= self.leaf_size:
            return KDTreeNode(
                True,
                level,
                X[:, :-1],
                X[:, -1].astype(int),
                None,
                None
            )
        dim = level % (X.shape[1] - 1)
        X = np.array(sorted(X, key=lambda x: x[dim]))
        median = len(X) // 2
        self.depth = max(self.depth, level + 1)
        return KDTreeNode(
            False,
            level,
            X[median, :-1].reshape(1, -1),
            X[median, -1].reshape(1).astype(int),
            self.kd_tree(X[:median, :], level + 1),
            self.kd_tree(X[median + 1:, :], level + 1)
        )

    @staticmethod
    def distances(p1, p2):
        return np.linalg.norm(p1 - p2, axis=1)

    def re_init(self, node):
        if node is not None:
            node.visited = False
            self.re_init(node.left)
            self.re_init(node.right)

    @staticmethod
    def visit_leaf(node: KDTreeNode, x: np.ndarray, heap: list):
        if not node.visited:
            dst = KDTreeOptimizer.distances(node.points, x)
            for d, i in zip(dst, node.indices):
                heapq.heappushpop(heap, (-d, i))
            node.visited = True
            if node.is_leaf:
                return node
        dim = node.level % len(x)
        # going down for the first time
        if not node.left.visited and not node.right.visited:
            if x[dim] <= node.points[0, dim]:
                return KDTreeOptimizer.visit_leaf(node.left, x, heap)
            else:
                return KDTreeOptimizer.visit_leaf(node.right, x, heap)
        # going down on backward pass
        elif not node.left.visited:
            return KDTreeOptimizer.visit_leaf(node.left, x, heap)
        elif not node.right.visited:
            return KDTreeOptimizer.visit_leaf(node.right, x, heap)
        else:
            return node

    def query(self, X: np.ndarray, k: int) -> np.ndarray:
        result = np.empty(X.shape[:-1] + (k,), dtype=np.uint32)
        for i, x in enumerate(X):
            self.re_init(self.root)
            heap = [(-np.inf, 0) for _ in range(k)]
            curr = KDTreeOptimizer.visit_leaf(self.root, x, heap)
            while curr.parent is not None:
                curr = curr.parent
                dim = curr.level % curr.points.shape[1]
                dst_to_line = abs(x[dim] - curr.points[0, dim])
                if dst_to_line <= -heap[0][0]:
                    curr = KDTreeOptimizer.visit_leaf(curr, x, heap)
            result[i] = [index for _, index in heapq.nlargest(k, heap)]
        return result
