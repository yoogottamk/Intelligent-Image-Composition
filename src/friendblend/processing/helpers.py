"""
Common functions used multiple times
"""

import cv2 as cv
import numpy as np


def draw_box(img, bounding_box):
    """
    Draws a bounding box in image
    """
    img = img.copy()
    x, y, w, h = bounding_box

    return cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 3)


def pt_in_box(pt, box):
    """
    pt: (x1, y1)
    box: (x, y, w, h)

    returns whether point is within box
    """
    x1, y1 = pt
    x, y, w, h = box
    return x <= x1 <= x + w and y <= y1 <= y + h


class DSU:
    """
    The union-find algorithm
    """

    def __init__(self, n):
        self.parent = np.array(list(range(n)))

    def find(self, i):
        """
        Finds the parent of i'th index
        """
        if self.parent[i] != i:
            self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i, j):
        """
        Puts i and j under the same subtree
        """
        x = self.find(i)
        y = self.find(j)

        if x == y:
            return

        if y < x:
            self.parent[x] = y
        else:
            self.parent[y] = x


def connected(img: np.ndarray) -> np.ndarray:
    """
    Connected components, implementing 4-connectivity
    """
    label_id = 1
    labels = np.zeros(img.shape[:2]).astype("int")
    nx, ny = img.shape[:2]

    for i in range(nx):
        for j in range(ny):
            if img[i, j] > 0:
                if i > 0 and labels[i - 1, j] != 0:
                    labels[i, j] = labels[i - 1, j]
                elif j > 0 and labels[i, j - 1] != 0:
                    labels[i, j] = labels[i, j - 1]
                else:
                    labels[i, j] = label_id
                    label_id += 1

    dsu = DSU(label_id + 1)

    for i in range(nx):
        for j in range(ny):
            if labels[i, j] != 0:
                if i > 0 and labels[i - 1, j] != labels[i, j] and labels[i - 1, j] != 0:
                    dsu.union(labels[i, j], labels[i - 1, j])
                elif (
                    j > 0 and labels[i, j - 1] != labels[i, j] and labels[i, j - 1] != 0
                ):
                    dsu.union(labels[i, j], labels[i, j - 1])

    for i in range(nx):
        for j in range(ny):
            if labels[i, j] != 0:
                labels[i, j] = dsu.find(labels[i, j])

    return labels
