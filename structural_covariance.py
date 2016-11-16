import numpy as np


class StructuralCovariance(object):

    def __init__(self):
        self.mean_roi = np.nan
        self.std_roi = np.nan

    def fit(self, X, y=None):
        X = self.standardize_subjects(X)
        self.mean_roi = X.mean(axis=0, keepdims=True)
        self.std_roi = X.std(axis=0, keepdims=True)

    def transform(self, X, y=None):
        n_subj, n_dims = X.shape
        X = self.standardize_subjects(X)
        X_standard = self.standardize_rois(X)
        ids_triu = np.triu_indices(n_dims, k=1)
        structural_covariance = np.zeros((n_subj, ids_triu[0].size))

        for id_subj in xrange(n_subj):
            subj_data = X_standard[id_subj]
            distances = (subj_data[:, np.newaxis] - subj_data[np.newaxis, :])
            structural_covariance[id_subj, :] = 1./np.exp(distances[ids_triu] ** 2)

        return structural_covariance

    def transform2(self, X, y=None):
        n_dims = X.shape[1]
        X = self.standardize_subjects(X)
        X = self.standardize_rois(X)
        ids_triu = np.triu_indices(n_dims, k=1)
        distances = X[:, :, np.newaxis] - X[:, np.newaxis, :]
        return np.array([1./np.exp(d[ids_triu] ** 2) for d in distances])

    def fit_transform(self, X, y=None):
        self.fit(X, y=None)
        return self.transform(X, y=None)

    @staticmethod
    def standardize_subjects(X):
        return (X - X.mean(axis=1, keepdims=True))/X.std(axis=1, keepdims=True)

    def standardize_rois(self, X):
        return (X - self.mean_roi)/self.std_roi
