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
        self.scaler = StandardScaler()
        self.scaler_fit = False
        self.gmm = GaussianMixture(*args, **kwargs)

    def _create_log_x(self, X):
        X = np.array(copy(X))
        for i in range(X.shape[1]):
            if i in self.logs:
                X[:, i] = np.log(X[:, i]+self.eps)
        if not self.scaler_fit:
            self.scaler.fit(X)
            self.scaler_fit = True
        X = self.scaler.transform(X)
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
        X = self.scaler.inverse_transform(X)
        for i in range(X.shape[1]):
            if i in self.logs:
                X[:, i] = np.exp(X[:, i])
        return X, comp