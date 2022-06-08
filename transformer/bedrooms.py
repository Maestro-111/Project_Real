
import re
import pandas as pd
from base.base_cfg import BaseCfg
from base.timer import Timer
from sklearn.base import BaseEstimator, TransformerMixin
from transformer.label_map import getLevel

logger = BaseCfg.getLogger(__name__)


class RmsTransformer(BaseEstimator, TransformerMixin):
    """rms transformer.

    Parameters
    ----------

    """
    masterReg = re.compile(
        r'^Master$|^Primary$|^Mbr$|^Prim\sBdrm$|^Master\s+B|^Primary\sB|', re.IGNORECASE)
    bedroomReg = re.compile(
        r'^\d+(nd|rd|th)?\s?(Br|Bedroom|B\/R|Bedro)\s?$|^(Bed|Bedrom|Bedroom|Br)\s?\d?$', re.IGNORECASE)

    def __init__(self):
        pass

    def target_cols(self):
        return ['rm_p_size_n', 'rm_p_area_n', 'rm_t_size_n', 'rm_t_area_n']

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
        logger.debug(f'transform rms')
        timer = Timer('rms', logger)
        timer.start()
        for col in self.target_cols():
            X[col] = 0
        for index, rms in X.loc[:, 'rms'].items():
            if not isinstance(rms, list):
                logger.warning(f'rms is not list: {rms}')
                continue
            totalSize = 0
            totalArea = 0
            for rm in rms:
                if not isinstance(rm, dict):
                    logger.warning(f'rm is not dict: {rm} {type(rm)}')
                    continue
                t = rm.get('t', None)
                if t is not None:
                    w = rm.get('w', 0)
                    h = rm.get('h', 0)
                    if self.masterReg.match(t):
                        X.loc[index, 'rm_p_size_n'] = w + h
                        X.loc[index, 'rm_p_area_n'] = w * h
                        totalSize += w + h
                        totalArea += w * h
                    elif self.bedroomReg.match(t) and getLevel(rm.get('l', 1)) >= 1:
                        totalSize += w + h
                        totalArea += w * h
            X.loc[index, 'rm_t_size_n'] = totalSize
            X.loc[index, 'rm_t_area_n'] = totalArea
        timer.stop()
        return X
