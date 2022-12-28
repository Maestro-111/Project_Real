

import math
from re import S

import pandas as pd
from base.base_cfg import BaseCfg
from base.mongo import getMongoClient
from base.timer import Timer
from base.const import DROP, Mode
from data.estimate_scale import EstimateScale
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from typing import Union
from scipy import stats

logger = BaseCfg.getLogger(__name__)


class DbNumericTransformer(TransformerMixin, BaseEstimator):
    """Transforms labels to integers and save the mapping to database

    Parameters
    ----------
    collection : mongodb collection name
        The mongodb collection to save the mapping to.
    col: str
        The feature name to save the mapping as.
    mode: Mode
        The mode of the transformer. PREDICT or TRAIN.
    na_value: str
        The value to fill the empty fields with. Can be DROP, MEAN, MEDIAN, MIN, MAX.

    Attributes
    ----------
    n_features_ : int
        The number of features of the data passed to :meth:`fit`.
    stats_ : dict
        The stats of the data passed to :meth:`fit`.
    """

    def __init__(
        self,
        collection: str,
        col: str,
        mode: Mode = Mode.TRAIN,
        na_value: Union[str, int, float] = None,
        est_scale: EstimateScale = None,
    ):
        self.collection = collection
        self.col = col
        self.mode = mode
        self.na_value = na_value
        self.est_scale = est_scale

    def get_feature_names(self):
        return [self.col+'-n']

    def set_params(self, **params):
        ret = super().set_params(**params)
        self._connect_db()
        return ret

    def _connect_db(self):
        if hasattr(self, '_db_connected') and self._db_connected:
            return
        if self.est_scale:
            self.est_scale_str = EstimateScale.to_str(self.est_scale) + ':'
        else:
            self.est_scale_str = ''
        self.db_id = f"{self.est_scale_str}{self.col}_n"
        self.stats = {}
        if self.mode == Mode.PREDICT:
            self.stats = getMongoClient().findOne(
                self.collection, {"_id": self.db_id})
        self._db_connected = True

    def _save_values(self, col_stats: dict):
        record = {**col_stats,
                  "_id": self.db_id,
                  "col": self.col,
                  "scale": self.est_scale_str,
                  }
        # print(record)
        getMongoClient().save(self.collection, record)

    def fit(self, X, y=None) -> 'DbNumericTransformer':
        """A reference implementation of a fitting function for a transformer.
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
        logger.debug(f'number {self.col} fit {self.mode}')
        # X = check_array(X, accept_sparse=True)

        # self.n_features_ = X.shape[1]

        # if self.n_features_ != 1:
        #     raise ValueError('Only one column is allowed')

        if self.mode == Mode.TRAIN:
            self._connect_db()
            t = Timer(self.col, logger)
            t.start()
            X = X.astype(float)
            # get mean, median, std, min, max, count
            describe = stats.describe(X, nan_policy='omit')
            mdeian = np.nanmedian(X)
            self.stats = {
                "mean": float(describe.mean),
                "median": mdeian,
                "min": float(describe.minmax[0]),
                "max": float(describe.minmax[1]),
                "std": math.sqrt(describe.variance),
                "count": int(describe.nobs),
                "variance": float(describe.variance),
                "skewness": float(describe.skewness),
                "kurtosis": float(describe.kurtosis),
            }
            self._save_values(self.stats)
            t.stop(X.shape[0])
        self.stats_ = self.stats
        # Return the transformer
        return self

    def transform(self, X, y=None) -> pd.DataFrame:
        """ A reference implementation of a transform function.
        Parameters
        ----------
        X : {array-like, sparse-matrix}, shape (n_samples, n_features)
            The input samples.
        Returns
        -------
        X_transformed : array, shape (n_samples, n_features)
            The array containing the element-wise square roots of the values
            in ``X``.
        """
        # Check is fit had been called
        # check_is_fitted(self, 'n_features_')

        # Input validation
        # X = check_array(X, accept_sparse=True)

        # Check that the input is of the same shape as the one passed
        # during fit.
        # if X.shape[1] != self.n_features_:
        #     raise ValueError('Shape of input is different from what was seen'
        #                      'in `fit`')

        # we may need to use a wrapper here
        # X = pd.DataFrame(X)
        logger.debug(f'number {self.col} transform {self.mode}')
        self._connect_db()
        if isinstance(self.na_value, str):
            if self.na_value == DROP:
                na_value = np.nan
            else:
                na_value = self.stats_[self.na_value]
        elif isinstance(self.na_value, int) or isinstance(self.na_value, float):
            na_value = self.na_value
        else:
            na_value = self.stats_["mean"]
        X.fillna(na_value, inplace=True)
        return X
        # return pd.DataFrame({self.col: X.values})
