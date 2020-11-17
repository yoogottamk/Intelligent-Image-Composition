"""
K Means Implementation
"""

import numpy as np
import matplotlib.pyplot as plt


class KMeans:
    def __init__(self, data, k=5, n_iters=2, DEBUG=False):
        """
        data: The points to apply K-Means on
              In this use-case, it is an array of pixel values (RGB)
        k: Number of clusters (also referred to as components in GMM)
        n_iters: Number of times to run the K-Means algorithm
        """
        # initializing necessary variables
        if DEBUG:
            self.data = data.reshape(-1, 2)
        else:
            self.data = data.reshape(-1, 3)
        self.n = self.data.shape[0]
        self.k = 5
        self.n_iters = n_iters
        self.clusters = np.zeros(self.n)

    def initialize_clusters(self):
        """
        initializing centers
        """
        inds = np.random.choice(self.n, self.k)
        self.centers = self.data[inds]
        while len(np.unique(self.centers, axis=1)) != self.k:
            inds = np.random.choice(self.n, self.k)
            self.centers = self.data[inds]

    def update_data_clusters(self):
        """
        Assigns the cluster based on distance from new cluster centers
        """
        i = 0
        for pixel in self.data:
            diffs = [np.sum((pixel - self.centers[j]) ** 2, 0) for j in range(self.k)]
            self.clusters[i] = np.argmin(np.array(diffs))
            i += 1

    def get_new_centers(self):
        """
        Update the centers 
        """
        for i in range(self.k):
            curr_cluster_points = self.data[np.where(self.clusters == i)]
            new_center = np.sum(curr_cluster_points, axis=0) / (
                1 + len(curr_cluster_points)
            )
            self.centers[i] = new_center

    def get_clusters(self):
        """
        Run K-Means algorithm to get clusters
        """
        self.initialize_clusters()
        for i in range(self.n_iters):
            self.update_data_clusters()
            self.get_new_centers()

        output_components = []
        for i in range(self.k):
            output_components += [self.data[np.where(self.clusters == i)]]
        return output_components

    def plot(self):
        """
        Plots the data
        """
        print(self.clusters)
        cols = ["r", "g", "b", "y", "orange"]
        for i in range(self.k):
            pixels = self.data[np.where(self.clusters == i)]
            plt.scatter(pixels[:, 0], pixels[:, 1], c=cols[i])

        plt.show()


if __name__ == "__main__":
    A = np.random.randint(0, 255, (5000, 2))
    k = KMeans(A, k=5, n_iters=2, DEBUG=True)
    k.get_clusters()
    k.plot()

