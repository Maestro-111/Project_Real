
from math import isnan
from base.base_cfg import BaseCfg

from base.timer import Timer
from numpy import nan
from sklearn.base import BaseEstimator, TransformerMixin
from transformer.db_label import DbLabelTransformer
from base.const import SAVE_LABEL_TO_DB, Mode

logger = BaseCfg.getLogger(__name__)


class OneHotArrayEncodingTransformer(DbLabelTransformer):
    """One hot encoding for field of string array.

    Parameters
    ----------
    col: str: the source column
    map: dict: the mapping from string to new feature string

    """

    def __init__(
        self,
        col: str,
        map: dict = None,
        sufix: str = '-b',
        collection: str = None,
        mode: Mode = Mode.TRAIN,
        na_value=None,
        save_to_db: bool = SAVE_LABEL_TO_DB,
    ):
        if collection is not None:
            super().__init__(collection, col, mode, na_value, save_to_db)
            self.col_category = self.col+'-c'
            # logger.info(
            #     f'init OneHotArrayEncodingTransformer with collection {self.col_category}')
        else:
            super().__init__(None, col, mode, na_value, save_to_db)
            self.col_category = None
        self.col = col
        self.sufix = sufix
        self._errors = {}
        self.map = {}
        if map is not None:
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
        if self.col_category is not None:
            retSet.add(self.col_category)
        for v in self.map.values():
            if isinstance(v, list):
                for vv in v:
                    retSet.add(self.col+'-'+vv)
            elif isinstance(v, str):
                retSet.add(self.col+'-'+v)
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
            logger.debug(f'fit {self.col} build map')
            col_index = X[self.col].value_counts().index
            col_values = [v + self.sufix for v in col_index]
            self.map = dict(zip(col_index, col_values))
            super().fit(X, y)
        elif self.col_category is not None:  # save map values to db
            logger.debug(f'fit {self.col} save label to db')
            Xs = X[self.col].apply(lambda x: self.map.get(x, x))
            super().fit(Xs, y)
            # print('self getattr labels-', getattr(self, 'labels-', 'None'))
            # print(f'self.col {self.col} in self.labels-',
            #       (self.col not in self.labels_))

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
        list_value_count = 0
        str_value_count = 0
        error_count = 0
        default_col_name = self.map.get('-', None)
        new_cols = self.target_cols()
        for col in new_cols:
            if col not in X.columns:
                X[col] = 0

        def _set_value(value, i):
            nonlocal error_count
            col_name = self.map.get(value, default_col_name)
            if isinstance(col_name, list):
                for v in col_name:
                    X.loc[i, (self.col+'-'+v)] = 1
            elif isinstance(col_name, str):
                X.loc[i, (self.col+'-'+col_name)] = 1
            elif value is None or (value == 'nan') or (value == '') or isnan(value):
                pass
            else:
                error_count += 1
                if self._errors.get(value, None) is not None:
                    self._errors[value] += 1
                    return
                self._errors[value] = 1
                logger.error(f'{self.col} {value} {type(value)} not in map')

        for i, row in X.iterrows():
            value = row[self.col]
            if (value is None) or (value is nan):
                continue
            if isinstance(value, list):
                for v in value:
                    _set_value(v, i)
                    list_value_count += 1
            else:
                _set_value(value, i)
                str_value_count += 1

        # set category column
        if self.col_category is not None:
            Xs = X[self.col].apply(lambda x: self.map.get(x, x))
            X.loc[:, self.col_category] = super().transform(Xs)

        logger.info(
            f'{self.col} list:{list_value_count} str:{str_value_count} error:{error_count}')
        t.stop(X.shape[0])
        return X
