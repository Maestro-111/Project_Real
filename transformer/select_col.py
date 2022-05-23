import pandas as pd
from base.base_cfg import BaseCfg
from sklearn.base import BaseEstimator, TransformerMixin

logger = BaseCfg.getLogger(__name__)


class SelectColumnTransformer(TransformerMixin, BaseEstimator):
    """Transforms string to 0 or 1

    Parameters
    ----------
    col: str : the target column name
    columns: list of strings of two or more: The columns to be selected.
    type: default int. The type of filter.

    Attributes
    ----------
    """

    def __init__(self, col: str, columns: list[str], v_type=int, as_na_value=None):
        self.col = col
        self.columns = columns
        self.v_type = v_type
        self.as_na_value = as_na_value

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
        logger.debug(f'fit {self.columns}')
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
        logger.debug(f'transform {self.columns}')

        def select_col(row):
            for col in self.columns:
                if type(row[col]) == self.v_type:
                    return row[col]
            return self.as_na_value
        X.loc[:, self.col] = X.apply(select_col, axis=1)
        return X
