
import pandas as pd
from base.base_cfg import BaseCfg
from sklearn.base import BaseEstimator, TransformerMixin

logger = BaseCfg.getLogger(__name__)


class BinaryTransformer(TransformerMixin, BaseEstimator):
    """Transforms string to 0 or 1

    Parameters
    ----------
    map: list of strings of two, or a function
        the list of values mapping to 0 and 1,
        or a function that return 0 or 1 based on the values

    Attributes
    ----------
    n_values_ : int
        The number of unique values of the data passed to :meth:`fit`.
    """

    def __init__(self, map, col: str):
        self.map = map
        self.col = col

    def fit(self, X, y=None):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        self : object
            Returns self.
        """
        logger.debug(f'binary.fit {self.col}')
        self.n_values_ = len(X.value_counts())
        return self

    def transform(self, X):
        """Transform X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        X_transformed : {array-like, sparse matrix}, shape (n_samples, n_features_)
            The transformed data.
        """
        logger.debug(f'binary.transform {self.col}')
        if isinstance(self.map, list):
            if len(self.map) == 2:
                X = [0 if x == self.map[0] else 1 for x in X]
            else:
                X = [0 if x is None else 1 for x in X]
        else:
            X = [0 if x is None else self.map(x) for x in X]
        # return X
        return pd.DataFrame({self.col: X})
