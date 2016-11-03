import numpy as np


class FeatureSelector(object):

    def __init__(self, z_thresh=3.5):
        self.z_thresh = z_thresh
        self.chosen_ftrs = np.nan

    def fit(self, X, y):

        y_class1 = y == 1
        y_class2 = y == 0

        mean_group1 = X[y_class1].mean(axis=0)
        mean_group2 = X[y_class2].mean(axis=0)

        mean_diffs = mean_group1 - mean_group2
        z_scores_diff = (mean_diffs - mean_diffs.mean())/mean_diffs.std()

        self.chosen_ftrs = np.abs(z_scores_diff) >= self.z_thresh

    def transform(self, X, y=None):
        return X[:, self.chosen_ftrs]

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X, y)