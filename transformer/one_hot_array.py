
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
        self.sufix = sufix
        self._errors = {}
        self.map = {}
        for (x, y) in map.items():
            if isinstance(y, list):
                for v in y:
                    self.map[x] = v+sufix
            elif isinstance(y, str):
                self.map[x] = y+sufix
            else:
                raise ValueError(f'{y} is not a string or list')

    def target_cols(self):
        retSet = set()
        for v in self.map.values():
            if isinstance(v, list):
                retSet.update(v)
            elif isinstance(v, str):
                retSet.add(v)
        return retSet

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
        default_col_name = self.map.get('_', None)
        new_cols = self.target_cols()
        for col in new_cols:
            if col not in X.columns:
                X[col] = 0
        for i, row in X.iterrows():
            value = row[self.col]
            if (value is None) or (value is nan):
                continue

            def _set_value(value, index):
                col_name = self.map.get(value, default_col_name)
                if isinstance(col_name, list):
                    for v in col_name:
                        X.loc[i, v] = 1
                elif isinstance(col_name, str):
                    X.loc[i, col_name] = 1
                else:
                    if self._errors.get(value, None) is not None:
                        self._errors[value] += 1
                        return
                    self._errors[value] = 1
                    logger.error(f'{self.col} {value} not in map')

            if isinstance(value, list):
                for v in value:
                    _set_value(v, i)
            else:
                _set_value(value, i)
        t.stop()
        return X
