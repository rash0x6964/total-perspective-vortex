import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class CSP(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=4):
        self.n_components = n_components
        self.filters_ = None
        self.eigenvalues_ = None

    def fit(self, X, y):
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError(
                f"CSP requires exactly 2 classes, got {len(classes)}: {classes}"
            )

        class1, class2 = classes

        C1 = self._mean_covariance(X[y == class1])
        C2 = self._mean_covariance(X[y == class2])

        C_total = C1 + C2

        P = self._whitening(C_total)

        S1 = P.T @ C1 @ P
        eigenvalues, W = np.linalg.eigh(S1)

        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        W = W[:, order]

        n = self.n_components // 2
        idx = np.concatenate([np.arange(n), np.arange(-n, 0)])
        W = W[:, idx]

        self.filters_ = P @ W
        self.eigenvalues_ = eigenvalues[idx]

        return self

    def transform(self, X):
        if self.filters_ is None:
            raise RuntimeError("CSP is not fitted yet. Call fit() first.")

        Z = np.tensordot(X, self.filters_, axes=([1], [0]))
        Z = Z.transpose(0, 2, 1)

        var = np.var(Z, axis=2)
        features = np.log(var + 1e-10)

        return features

    def _mean_covariance(self, X):
        n_epochs = X.shape[0]
        n_channels = X.shape[1]
        C_sum = np.zeros((n_channels, n_channels))

        for epoch in X:
            cov = epoch @ epoch.T
            tr = np.trace(cov)
            C_sum += cov / tr

        return C_sum / n_epochs

    def _whitening(self, C):
        eigenvalues, eigenvectors = np.linalg.eigh(C)
        eigenvalues = np.maximum(eigenvalues, 1e-10)
        P = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T

        return P
