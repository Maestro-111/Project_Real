
from math import isnan
import re
import pandas as pd
from base.base_cfg import BaseCfg
from base.timer import Timer
from sklearn.base import BaseEstimator, TransformerMixin
from transformer.const_label_map import getLevel

logger = BaseCfg.getLogger(__name__)


class RmsTransformer(BaseEstimator, TransformerMixin):
    """rms transformer.
    transform rms to rms-p_size-n, rms-p_area-n, rms-t_size-n, rms-t_area-n

    Parameters
    ----------
    None
    """
    masterReg = re.compile(
        r'^Master$|^Primary$|^Mbr$|^Prim\sBdrm$|^Master\s+B|^Primary\sB|', re.IGNORECASE)
    bedroomReg = re.compile(
        r'^\d+(nd|rd|th)?\s?(Br|Bedroom|B\/R|Bedro)\s?$|^(Bed|Bedrom|Bedroom|Br)\s?\d?$', re.IGNORECASE)

    def __init__(self):
        pass

    def get_feature_names_out(self):
        return ['rms-p_size-n', 'rms-p_area-n', 'rms-t_size-n', 'rms-t_area-n']

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
        # logger.debug(f'fit rms')
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
        # logger.debug(f'transform rms')
        timer = Timer('rms', logger)
        timer.start()
        nanCount = 0
        totalCount = 0
        new_cols = self.get_feature_names_out()
        X[new_cols] = 0
        if 'rms' not in X.columns:
            logger.warning('rms not in X.columns')
            X['rms'] = [None for _ in range(len(X))]
        else:
            for index, rms in X.loc[:, 'rms'].items():
                totalCount += 1
                if not isinstance(rms, list):
                    if isnan(rms):
                        nanCount += 1
                        continue
                    logger.warning(f'rms is not list: {rms}')
                    continue
                totalSize = 0
                totalArea = 0
                for rm in rms:
                    if not isinstance(rm, dict):
                        if isnan(rm):
                            continue
                        logger.warning(f'rm is not dict: {rm} {type(rm)}')
                        continue
                    t = rm.get('t', None)
                    if t is not None:
                        w = rm.get('w', 0) or 0
                        h = rm.get('h', 0) or 0
                        if self.masterReg.match(t):
                            X.loc[index, 'rms-p_size-n'] = w + h
                            X.loc[index, 'rms-p_area-n'] = w * h
                            totalSize += w + h
                            totalArea += w * h
                        elif self.bedroomReg.match(t) and getLevel(rm.get('l', 1)) >= 1:
                            totalSize += w + h
                            totalArea += w * h
                X.loc[index, 'rms-t_size-n'] = totalSize
                X.loc[index, 'rms-t_area-n'] = totalArea
        if nanCount > 0:
            logger.warning(f'{nanCount}/{totalCount} nan values in rms')
        timer.stop(totalCount)
        return X
