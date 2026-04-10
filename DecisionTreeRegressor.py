import numpy as np

class DecisionTreeRegressor():
    def __init__(self, max_depth, min_samples_split, min_samples_leaf):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

        self.curr_depth = None

    def fit(self, X):
        if self.curr_depth == self.max_depth:
            return True