
import datetime
from math import isnan
import re
import pandas as pd
from base.base_cfg import BaseCfg
from base.timer import Timer
from base.util import dateFromNum
from sklearn.base import BaseEstimator, TransformerMixin
from transformer.label_map import getLevel

logger = BaseCfg.getLogger(__name__)
DATE_20000101 = datetime.datetime(2000, 1, 1, 0, 0)


class DatesTransformer(BaseEstimator, TransformerMixin):
    """bths transformer.

    Parameters
    ----------

    """
    maxLevel = 3

    def __init__(self, cols):
        self._target_cols = [
            'onD_2k_n',
            'onD_year_n',
            'onD_month_n',
            'onD_week_n'
        ]
        if 'offD' in cols:
            self.offD = True
            self._target_cols.append('offD_2k_n')
            self._target_cols.append('dom_n')
        else:
            self.offD = False
        if 'sldd' in cols:
            self.sldd = True
            self._target_cols.append('sldd_n')
        else:
            self.sldd = False

    def target_cols(self):
        return self._target_cols

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
        logger.debug(f'fit dates')
        if 'sldd' in X.columns:
            self.sldd = True
        if 'offD' in X.columns:
            self.offD = True
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
        logger.debug(f'transform dates')
        timer = Timer('bths', logger)
        nanCount = 0
        offDNanCount = 0
        totalCount = 0
        for col in self.target_cols():
            X[col] = None
        for i, row in X.iterrows():
            totalCount += 1
            onD = dateFromNum(row['onD'])
            if onD is not None:
                X.loc[i, 'onD_2k_n'] = abs((onD - DATE_20000101).days)
                X.loc[i, 'onD_year_n'] = onD.year
                X.loc[i, 'onD_month_n'] = onD.month
                X.loc[i, 'onD_week_n'] = onD.isocalendar().week
                if self.offD:
                    offD = dateFromNum(row['offD'])
                    if offD is not None:
                        X.loc[i, 'offD_2k_n'] = abs(
                            (offD - DATE_20000101).days)
                        X.loc[i, 'dom_n'] = abs((offD - onD).days)
                    else:
                        offDNanCount += 1
            else:
                nanCount += 1
            if self.sldd:
                sldd = dateFromNum(row['sldd'])
                if sldd is not None:
                    X.loc[i, 'sldd_n'] = abs((sldd - DATE_20000101).days)
        if nanCount > 0:
            logger.warn(f'onD is None: {nanCount}/{totalCount}')
        if offDNanCount > 0:
            logger.info(f'offD is None: {offDNanCount}/{totalCount}')
        timer.stop(totalCount)
        return X
