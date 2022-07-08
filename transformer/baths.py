
from math import isnan
import re
import pandas as pd
from base.base_cfg import BaseCfg
from base.timer import Timer
from sklearn.base import BaseEstimator, TransformerMixin
from transformer.label_map import getLevel

logger = BaseCfg.getLogger(__name__)


class BthsTransformer(BaseEstimator, TransformerMixin):
    """bths transformer.

    Parameters
    ----------

    """
    maxLevel = 3

    def __init__(self):
        _target_cols = []
        for n in range(BthsTransformer.maxLevel+1):  # 0-3
            _target_cols.append(f'bths-t{n}-n')
            _target_cols.append(f'bths-pc{n}-n')
        self._target_cols = _target_cols

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
        logger.debug(f'fit rms')
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
        logger.debug(f'transform baths')
        timer = Timer('bths', logger)
        timer.start()
        nanCount = 0
        totalCount = 0
        for col in self.target_cols():
            X[col] = 0
        for index, bths in X.loc[:, 'bths'].items():
            totalCount += 1
            if not isinstance(bths, list):
                if isnan(bths):
                    nanCount += 1
                    continue
                logger.warning(f'bths is not list: {bths}')
                continue
            for bth in bths:
                if not isinstance(bth, dict):
                    if isnan(bth):
                        continue
                    logger.warning(f'bth is not dict: {bth} {type(bth)}')
                    continue
                level = max(
                    0, min(round(getLevel(bth.get('l', 1)) + 0.01), BthsTransformer.maxLevel))
                t = bth.get('t', 0)
                X.loc[index, f'bths-t{level}-n'] += t
                X.loc[index, f'bths-pc{level}-n'] += t * bth.get('p', 0)
        if nanCount > 0:
            logger.warning(f'{nanCount}/{totalCount} nan values in bths')
        timer.stop(X.shape[0])
        return X
