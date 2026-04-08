import numpy as np
import math

class DBSCAN():
    def __init__(self, eps, min_samples):
        self.eps = eps
        self.min_samples = min_samples

    def fit(self, X):
        n = len(X)
        clusters = 0
        visited = -np.ones(n)
        labels = -np.ones(n)

        for p in range(n):
            visited[p] = 1

            N = self.neighborhood(X[p], X)

            if len(N) < self.min_samples:
                labels[p] = -1
                break

            clusters += 1

            self.clusterExpand(X[p], )

    
    def neighborhood(self, p, X):
        N = []

        for i in range(X):
            if math.dist(p, X[i]) < self.eps:
                N.append(X[i])
        
        return N

    def clusterExpand(self, p, N, cluster, X):
        return True
