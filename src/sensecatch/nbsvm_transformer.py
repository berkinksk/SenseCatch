"""NB log-count-ratio transformer for the deployed NBSVM model (Step 7.7).

Lives in src/sensecatch/ so the app can import it when unpickling models/nbsvm.pkl
(the deployed NBSVM's text "vectorizer" is a Pipeline: binary CountVectorizer -> this).
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class NBLogCountRatio(BaseEstimator, TransformerMixin):
    """Multiply binarized count features by the NB log-count-ratio r, estimated from TRAIN labels.

    Wang & Manning (2012): r = log( (p/||p||_1) / (q/||q||_1) ), p/q = smoothed positive/negative
    feature counts. Use inside a Pipeline AFTER a binary CountVectorizer; .fit needs y.
    """

    def fit(self, X, y):
        y = np.asarray(y)
        p = 1.0 + np.asarray(X[y == 1].sum(axis=0)).ravel()
        q = 1.0 + np.asarray(X[y == 0].sum(axis=0)).ravel()
        self.r_ = np.log((p / p.sum()) / (q / q.sum())).reshape(1, -1)
        return self

    def transform(self, X):
        return X.multiply(self.r_).tocsr()
