from matplotlib.transforms import Transform
from base.base_cfg import BaseCfg
from base.timer import Timer
from numpy import nan
from sklearn.base import BaseEstimator, TransformerMixin

logger = BaseCfg.getLogger(__name__)


class OneHotArrayEncodingTransformer(BaseEstimator, TransformerMixin):
    """One hot encoding for field of string array.

    Parameters
    ----------
    col: str: the source column
    map: dict: the mapping from string to new feature string

    """

    def __init__(self, col: str, map: dict = None, sufix: str = '_b'):
        self.col = col
        self.map = {x: y+sufix for (x, y) in map.items()}
        self.sufix = sufix
        self._errors = {}

    def target_cols(self):
        return set(self.map.values())

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
        logger.debug(f'fit {self.col}')
        if self.map is None:
            col_index = X[self.col].value_counts().index
            col_values = [v + self.sufix for v in col_index]
            self.map = dict(zip(col_index, col_values))
        self.n_col_ = len(self.target_cols())
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
        logger.debug(f'transform {self.col}')
        t = Timer(self.col, logger)
        t.start()
        new_cols = self.target_cols()
        for col in new_cols:
            X[col] = 0
        for i, row in X.iterrows():
            value = row[self.col]
            if (value is None) or (value is nan):
                continue
            try:
                if isinstance(value, list):
                    for v in value:
                        X.loc[i, self.map[v]] = 1
                else:
                    X.loc[i, self.map[value]] = 1
            except KeyError:
                if self._errors[value]:
                    self._errors[value] += 1
                    continue
                self._errors[value] = 1
                logger.error(f'{self.col} {value} not in {self.map}')
        t.stop()
        return X
