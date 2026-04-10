import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

class DBSCAN():

    def __init__(self, eps, min_samples):
        self.eps = eps
        self.min_samples = min_samples

        self.labels = None

    def fit(self, X):
        n = len(X)
        cluster = 0
        self.labels = -np.ones(n)

        for i in range(n):
            # Ignores points already visited
            if self.labels[i] != -1:
                continue

            neighbors = self.get_neighborhood(X[i], X)

            # Label as noise or expand core point
            if len(neighbors) < self.min_samples:
                self.labels[i] = -2
            else:
                self.expand_cluster(X, i, neighbors, cluster)
                cluster += 1

        return self

   

    def get_neighborhood(self, p, X):
        n, neighbors = len(X), []

        for i in range(n):
            if math.dist(p, X[i]) <= self.eps:
                neighbors.append(i)

        return neighbors

    def expand_cluster(self, X, i, neighbors, cluster):
        self.labels[i], idx = cluster, 0

        while idx < len(neighbors):
            curr_p = neighbors[idx]

            if self.labels[curr_p] == -1:
                self.labels[curr_p] = cluster
                new_neighbors = self.get_neighborhood(X[curr_p], X)

                if len(new_neighbors) >= self.min_samples:
                    for n_idx in new_neighbors:
                        if n_idx not in neighbors:
                            neighbors.append(n_idx)

            idx += 1

if __name__ == "__main__":
    X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

    model = DBSCAN(eps=0.6, min_samples=4)
    model.fit(X)

    unique_labels = set(model.labels)
    colors = plt.cm.get_cmap("Spectral")(np.linspace(0, 1, len(unique_labels)))

    for k, col in zip(unique_labels, colors):
        if k == -2:
            col = [0, 0, 0, 1]

        class_member_mask = (model.labels == k)
        xy = X[class_member_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=6)

    plt.title("DBSCAN: Clusters Identificados e Ruído (Preto)")
    plt.show()