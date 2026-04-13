import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler

class DBSCAN():

    def __init__(self, eps, min_samples, metrics):
        assert metrics in ["euclidian", "manhattan"], "Métrica inválida"

        self.eps = eps
        self.min_samples = min_samples
        self.metrics = metrics

        self.labels = None

    def fit_predict(self, X):
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
        if self.metrics == "euclidian":
            distances = np.linalg.norm(X - p, axis=1)
        elif self.metrics == "manhattan":
            distances = np.sum(np.abs(X - p), axis=1)
        
        neighbors = np.where(distances <= self.eps)[0]
    
        return neighbors.tolist()

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
    X, y_true = make_moons(n_samples=300, noise=0.05, random_state=0)
    
    X = StandardScaler().fit_transform(X)

    model = DBSCAN(eps=0.3, min_samples=5, metrics="euclidian")
    model.fit_predict(X)

    unique_labels = set(model.labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

    for k, col in zip(unique_labels, colors):
        if k == -2:
            col = [0, 0, 0, 1]

        class_member_mask = (model.labels == k)
        xy = X[class_member_mask]
        plt.scatter(xy[:, 0], xy[:, 1], color=tuple(col), edgecolor='k', s=50)

    plt.title(f"DBSCAN - Clusters: {len(unique_labels) - (1 if -2 in unique_labels else 0)}")
    plt.show()