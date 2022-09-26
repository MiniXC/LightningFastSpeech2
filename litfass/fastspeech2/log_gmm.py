from sklearn.mixture import GaussianMixture
import numpy as np
from copy import copy
from sklearn.preprocessing import StandardScaler

# TODO: add standard scaler

class LogGMM():
    def __init__(self, *args, **kwargs):
        if "logs" in kwargs:
            self.logs = kwargs["logs"]
            del kwargs["logs"]
        else:
            self.logs = []
        if "eps" in kwargs:
            self.eps = kwargs["eps"]
            del kwargs["eps"]
        else:
            self.eps = 1e-10
        self.max_vals = None
        self.gmm = GaussianMixture(*args, **kwargs)

    def _create_log_x(self, X):
        X = np.array(copy(X))
        if self.max_vals is None:
            self.max_vals = np.max(X, axis=0)
        X = X / self.max_vals + self.eps
        for i in range(X.shape[1]):
            if i in self.logs:
                X[:, i] = np.log(X[:, i])
        return X

    def fit(self, X, y=None):
        X = self._create_log_x(X)
        return self.gmm.fit(X, y)

    def fit_predict(self, X, y=None):
        X = self._create_log_x(X)
        return self.gmm.fit_predict(X, y)

    def predict(self, X):
        X = self._create_log_x(X)
        return self.gmm.predict(X)

    def predict_proba(self, X):
        X = self._create_log_x(X)
        return self.gmm.predict_proba(X)

    def score_samples(self, X):
        X = self._create_log_x(X)
        return self.gmm.score_samples(X)

    def score(self, X, y=None):
        X = self._create_log_x(X)
        return self.gmm.score(X, y)

    def bic(self, X):
        X = self._create_log_x(X)
        return self.gmm.bic(X)

    def aic(self, X):
        X = self._create_log_x(X)
        return self.gmm.aic(X)

    def sample(self, n_samples=1, random_state=None):
        np.random.seed(random_state)
        X, comp = self.gmm.sample(n_samples)
        for i in range(X.shape[1]):
            if i in self.logs:
                X[:, i] = (np.exp(X[:, i])-self.eps)*self.max_vals[i]
            else:
                X[:, i] = (X[:, i]-self.eps)*self.max_vals[i]
        return X, comp