import os
import re
import glob
import operator
from IPython import display
from PIL import Image
from .optimizer import Optimizer
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
        if len(X) <= self.leaf_size or self.leaf_size == 1 and len(X) == 2:
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
    def visit_leaf(node: KDTreeNode, x: np.ndarray, heap: list, k: int):
        if not node.visited:
            dst = KDTreeOptimizer.distances(node.points, x)
            if len(dst) > k:
                nearest_indices = np.argpartition(dst, k - 1)[:k]
                new_dst = dst[nearest_indices]
                indices = node.indices[nearest_indices]
            else:
                new_dst = dst
                indices = node.indices
            for d, i in zip(new_dst, indices):
                heapq.heappushpop(heap, (-d, i))
            node.visited = True
            if node.is_leaf:
                return node
        dim = node.level % len(x)
        # going down for the first time
        if not node.left.visited and not node.right.visited:
            if x[dim] <= node.points[0, dim]:
                return KDTreeOptimizer.visit_leaf(node.left, x, heap, k)
            else:
                return KDTreeOptimizer.visit_leaf(node.right, x, heap, k)
        # going down on backward pass
        elif not node.left.visited:
            return KDTreeOptimizer.visit_leaf(node.left, x, heap, k)
        elif not node.right.visited:
            return KDTreeOptimizer.visit_leaf(node.right, x, heap, k)
        else:
            return node

    def query(self, X: np.ndarray, k: int) -> np.ndarray:
        result = np.empty(X.shape[:-1] + (k,), dtype=np.uint32)
        for i, x in enumerate(X):
            self.re_init(self.root)
            heap = [(-np.inf, 0) for _ in range(k)]
            curr = KDTreeOptimizer.visit_leaf(self.root, x, heap, k)
            while curr.parent is not None:
                curr = curr.parent
                dim = curr.level % curr.points.shape[1]
                dst_to_line = abs(x[dim] - curr.points[0, dim])
                if dst_to_line <= -heap[0][0]:
                    curr = KDTreeOptimizer.visit_leaf(curr, x, heap, k)
            result[i] = np.array(heap)[:, 1]
        return result

    @staticmethod
    def prepare_animation(folder='animate/kd_tree'):
        """clean animation snapshots directory"""
        images = glob.glob(f"{folder}/*.png")
        for file in images:
            os.remove(file)

    @staticmethod
    def display_animation(fig: plt.Figure, folder='animate/kd_tree'):
        """shows gif in-place"""
        plt.close(fig)

        frames = []
        images = glob.glob(f"{folder}/*.png")

        # extract image file names as numbers
        order = list(map(int, [re.search(r'\d+', s)[0] for s in images]))
        # sort images by their names as numbers
        pairs = sorted(zip(images, order), key=operator.itemgetter(1))

        # create frames
        frames = [Image.open(file) for file, _ in pairs]

        # Save into a GIF file that loops forever
        frames[0].save('gifs/KDTreeConstruction.gif',
                       format='GIF', loop=0,
                       append_images=frames[1:],
                       save_all=True, duration=1500)

        # display gif
        with open('gifs/KDTreeConstruction.gif', 'rb') as file:
            display.display(display.Image(file.read()))

    def draw_kd_tree(self, node, xlim, ylim, axis, bbox):
        if node is None or node.is_leaf:
            return
        (xmin, xmax), (ymin, ymax) = bbox
        med = node.points[0, axis]
        if axis == 0:
            ymin_norm = (ylim[0] - ymin) / (ymax - ymin)
            ymax_norm = (ylim[1] - ymin) / (ymax - ymin)
            plt.axvline(med, ymin=ymin_norm, ymax=ymax_norm, color='blue')
            self.draw_kd_tree(node.left, (xlim[0], med), ylim, 1, bbox)
            self.draw_kd_tree(node.right, (med, xlim[1]), ylim, 1, bbox)
        else:
            xmin_norm = (xlim[0] - xmin) / (xmax - xmin)
            xmax_norm = (xlim[1] - xmin) / (xmax - xmin)
            plt.axhline(med, xmin=xmin_norm, xmax=xmax_norm, color='red')
            self.draw_kd_tree(node.left, xlim, (ylim[0], med), 0, bbox)
            self.draw_kd_tree(node.right, xlim, (med, ylim[1]), 0, bbox)

    def visualize(self):
        plt.figure(figsize=(12, 12))
        plt.scatter(*self.X.T, color='black')
        xlim = (self.X[:, 0].min(), self.X[:, 0].max())
        ylim = (self.X[:, 1].min(), self.X[:, 1].max())
        plt.xlim(xlim)
        plt.ylim(ylim)
        self.draw_kd_tree(self.root, xlim, ylim, 0, (xlim, ylim))

    def visualize_query(self, point, k):
        self.visualize()
        plt.scatter(*point[0], color='red')
        nearest = self.query(point, k)
        plt.scatter(*self.X[nearest[0]].T, color='orange')

    def plot(self, animate=False, show_visited=False, show_level=None, fig_ax=None):
        if animate:
            KDTreeOptimizer.prepare_animation()
        # create figure
        if fig_ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
        else:
            fig, ax = fig_ax
        # draw data points
        ax.scatter(*self.X.T, color='black')

        # set up axes
        center = np.mean(self.X, axis=0)
        radius = max(np.max(self.X, axis=0) - center)
        offset_ratio = 0.15
        xoffset = (self.X[:, 0].max() - self.X[:, 0].min()) * offset_ratio
        yoffset = (self.X[:, 1].max() - self.X[:, 1].min()) * offset_ratio
        x_bord = (center[0] - radius - xoffset, center[0] + radius + xoffset)
        y_bord = (center[1] - radius - yoffset, center[1] + radius + yoffset)
        (xmin, xmax), (ymin, ymax) = x_bord, y_bord
        ax.set_xlim(x_bord)
        ax.set_ylim(y_bord)
        ax.set_xticks([])
        ax.set_yticks([])

        # prepare variables
        prev_node = None
        queue = [(self.root, x_bord, y_bord)]

        # perform BFS throughout nodes to create balls
        for node, xlim, ylim in queue:
            if not node.is_leaf:
                axis = node.level % 2
                med = node.points[0, axis]
                if show_level is None or node.level == show_level:
                    # if we enter new level, save figure before drawing new circle
                    if animate and (prev_node is None or node.level != prev_node.level):
                        fig.tight_layout()
                        fig.savefig(f'animate/kd_tree/{node.level}.png')
                        prev_node = node

                    # we will separately create ball filling and boundary
                    if axis == 0:
                        ymin_norm = (ylim[0] - ymin) / (ymax - ymin)
                        ymax_norm = (ylim[1] - ymin) / (ymax - ymin)
                        plt.axvline(med, ymin=ymin_norm, ymax=ymax_norm, color='blue')
                    else:
                        xmin_norm = (xlim[0] - xmin) / (xmax - xmin)
                        xmax_norm = (xlim[1] - xmin) / (xmax - xmin)
                        plt.axhline(med, xmin=xmin_norm, xmax=xmax_norm, color='red')
                # update queue
                if axis == 0:
                    queue.append((node.left, (xlim[0], med), ylim))
                    queue.append((node.right, (med, xlim[1]), ylim))
                else:
                    queue.append((node.left, xlim, (ylim[0], med)))
                    queue.append((node.right, xlim, (med, ylim[1])))
            elif show_visited and node.visited:
                rect = plt.Rectangle((xlim[0], ylim[0]), xlim[1] - xlim[0], ylim[1] - ylim[0], color='gray', alpha=0.4)
                ax.add_patch(rect)


        # disaply animation
        if animate:
            fig.savefig(f'animate/kd_tree/{self.depth + 1}.png')
            KDTreeOptimizer.display_animation(fig)
