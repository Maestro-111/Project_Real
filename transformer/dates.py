
import datetime
from math import isnan
import re
import pandas as pd
from base.base_cfg import BaseCfg
from base.timer import Timer
from base.util import dateFromNum
from sklearn.base import BaseEstimator, TransformerMixin
from transformer.const_label_map import getLevel

logger = BaseCfg.getLogger(__name__)
DATE_20000101 = datetime.datetime(2000, 1, 1, 0, 0)


class DatesTransformer(BaseEstimator, TransformerMixin):
    """Dates transformer.
    Transform dates to numeric values.

    Parameters
    ----------

    """
    maxLevel = 3

    def __init__(
        self,
        cols
    ):
        self.cols = cols

    def get_feature_names_out(self, cols: list[str] = None):

        if hasattr(self, '_target_cols') and self._target_cols is not None:
            return self._target_cols

        self._target_cols = [
            # 'onD-2k-n',
            'onD-year-n',
            'onD-season-n',
            'onD-month-n',  # based on Janurary of previous year, could be negative
            # 'onD-week-n',
            # 'onD-dayOfWeek-n',
            # 'onD-dayOfMonth-n',
            # 'onD-dayOfYear-n',
        ]
        if cols is None:
            cols = self.cols
        if 'offD' in cols:
            self.offD = True
            # self._target_cols.append('offD-2k-n')
            # self._target_cols.append('dom-n')
        else:
            self.offD = False
        if 'sldd' in cols:
            self.sldd = True
            # self._target_cols.append('sldd-dom-n')
        else:
            self.sldd = False
        return self._target_cols

    def set_params(self, **params):
        ret = super().set_params(**params)
        self._target_cols = None
        return ret

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
        # logger.debug(f'fit dates')

        self.get_feature_names_out()
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
        # logger.debug(f'transform dates')
        timer = Timer(self.__class__.__name__, logger)
        nanCount = 0
        offDNanCount = 0
        totalCount = 0
        toAddCols = [col for col in self.get_feature_names_out()
                     if col not in X.columns]
        X = pd.concat(
            [X, pd.DataFrame(columns=toAddCols, index=X.index)], axis=1)
        # hasOffD = 'offD' in X.columns
        # hasSldd = 'sldd' in X.columns
        previousYear = datetime.datetime.now().year - 1
        for i, row in X.iterrows():
            totalCount += 1
            onD = dateFromNum(row['onD'])
            if onD is not None:
                X.loc[i, 'onD-year-n'] = onD.year
                X.loc[i, 'onD-season-n'] = onD.month // 3
                X.loc[i, 'onD-month-n'] = onD.month + \
                    (onD.year - previousYear) * 12
                # X.loc[i, 'onD-week-n'] = onD.isocalendar().week
                #X.loc[i, 'onD-2k-n'] = abs((onD - DATE_20000101).days)
                # X.loc[i, 'onD-dayOfWeek-n'] = onD.weekday()
                # X.loc[i, 'onD-dayOfMonth-n'] = onD.day
                # X.loc[i, 'onD-dayOfYear-n'] = onD.timetuple().tm_yday
                # if hasOffD:
                #     offD = dateFromNum(row['offD'])
                #     if offD is not None:
                #         # X.loc[i, 'offD-2k-n'] = abs(
                #         #     (offD - DATE_20000101).days)
                #         X.loc[i, 'dom-n'] = abs((offD - onD).days)
                #     else:
                #         offDNanCount += 1
            else:
                nanCount += 1
                logger.warn(f'onD is None: {row}')
            # if hasSldd:
            #     sldd = dateFromNum(row['sldd'])
            #     if sldd is not None:
            #         X.loc[i, 'sldd-dom-n'] = abs((sldd - onD).days)
        if nanCount > 0:
            logger.warn(f'onD is None: {nanCount}/{totalCount}')
        # if offDNanCount > 0:
        #     logger.info(f'offD is None: {offDNanCount}/{totalCount}')
        timer.stop(totalCount)
        return X
