import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class LDA(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.classes_ = None
        self.means_ = None
        self.priors_ = None
        self.coef_ = None
        self.intercept_ = None
        self.S_within_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        n_samples, n_features = X.shape
        n_classes = len(self.classes_)

        overall_mean = X.mean(axis=0)

        self.means_ = np.array([
            X[y == c].mean(axis=0)
            for c in self.classes_
        ])

        self.priors_ = np.array([
            np.sum(y == c) / n_samples
            for c in self.classes_
        ])

        S_within = np.zeros((n_features, n_features))
        for c, mean_c in zip(self.classes_, self.means_):
            X_c = X[y == c] - mean_c
            S_within += X_c.T @ X_c

        S_between = np.zeros((n_features, n_features))
        for c, mean_c, prior in zip(self.classes_, self.means_, self.priors_):
            n_c = np.sum(y == c)
            diff = (mean_c - overall_mean).reshape(-1, 1)
            S_between += n_c * (diff @ diff.T)

        S_within_inv = np.linalg.inv(S_within)

        eigenvalues, eigenvectors = np.linalg.eig(S_within_inv @ S_between)

        eigenvalues = eigenvalues.real
        eigenvectors = eigenvectors.real

        order = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, order]

        self.coef_ = eigenvectors[:, 0]

        projected_means = self.means_ @ self.coef_
        self.intercept_ = -np.dot(self.priors_, projected_means)

        self.S_within_ = S_within

        return self

    def predict(self, X):
        scores = X @ self.coef_ + self.intercept_

        projected_means = self.means_ @ self.coef_

        pos_class = self.classes_[np.argmax(projected_means)]
        neg_class = self.classes_[np.argmin(projected_means)]

        return np.where(scores > 0, pos_class, neg_class)

    def predict_proba(self, X):
        scores = X @ self.coef_ + self.intercept_
        prob_1 = 1.0 / (1.0 + np.exp(-scores))
        return np.column_stack([1 - prob_1, prob_1])


if __name__ == "__main__":
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from csp import CSP
    import sklearn.discriminant_analysis as sklearn_lda

    np.random.seed(42)
    n_epochs, n_channels, n_times = 100, 64, 641
    X = np.random.randn(n_epochs, n_channels, n_times)
    y = np.array([1] * 50 + [2] * 50)

    X[y == 1, 0, :] += 3.0
    X[y == 2, 1, :] += 3.0

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # custom LDA
    pipeline_custom = Pipeline([
        ('csp', CSP(n_components=4)),
        ('clf', LDA())
    ])
    scores_custom = cross_val_score(pipeline_custom, X, y, cv=cv)
    print(f"Custom  LDA: {scores_custom.mean():.4f}")

    # sklearn LDA
    pipeline_sklearn = Pipeline([
        ('csp', CSP(n_components=4)),
        ('clf', sklearn_lda.LinearDiscriminantAnalysis())
    ])
    scores_sklearn = cross_val_score(pipeline_sklearn, X, y, cv=cv)
    print(f"Sklearn LDA: {scores_sklearn.mean():.4f}")
