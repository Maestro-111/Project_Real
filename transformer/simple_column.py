
from base.base_cfg import BaseCfg
from base.const import CONCURRENT_PROCESSES_MAX
from base.timer import Timer
from base.util import addColumns, getMemoryLimitedExtraProcessNumber
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import psutil
import numpy as np
import concurrent.futures

MULTI_PROCESS_THRESHOLD = 10000

logger = BaseCfg.getLogger(__name__)


class SimpleColumnTransformer(TransformerMixin, BaseEstimator):
    """Column Transformer.
    Perform column transform based on a list of column names.

    Parameters
    __________
    transFunctions: list(tuple)
        tuple of (name, function, sourceCol, targetCol)

    """

    def __init__(
        self,
        transFunctions: list[tuple],
        num_procs: int = CONCURRENT_PROCESSES_MAX
    ):
        self.transFunctions = transFunctions
        self.num_procs = num_procs

    def _build_internal_trans_func(self, force=False):
        if not force and hasattr(self, 'transFunctions_'):
            return self.transFunctions_
        self.transFunctions_ = []
        self.transFunctionsByName_ = {}
        for trans in self.transFunctions:
            trans_signature = len(trans)
            if trans_signature == 2:
                self.transFunctions_.append((*trans, None, None, False))
            elif trans_signature == 3:
                self.transFunctions_.append((*trans, trans[2], False))
            elif trans_signature == 4:
                self.transFunctions_.append((*trans, False))
            elif trans_signature == 5:
                self.transFunctions_.append(trans)
            else:
                raise ValueError(
                    f'Invalid number of parameters in list(tuple) {trans_signature}')
            self.transFunctionsByName_[trans[0]] = trans
        return self.transFunctions_

    def get_feature_names_out(self) -> list[str]:
        self._target_cols = []
        self._build_internal_trans_func()
        for name, func, col, targetCol, preTransform in self.transFunctions_:
            addedCols = False
            if targetCol:
                self._target_cols.append(targetCol)
                addedCols = True
            if hasattr(func, 'get_feature_names_out'):
                cols = func.get_feature_names_out()
                if cols is not None:
                    self._target_cols.extend(cols)
                    addedCols = True
            if not addedCols:
                logger.warning(
                    f'No targetCol or get_feature_names_out for {name} input column {col}')
        return self._target_cols

    def fit(self, X, y=None):
        self._build_internal_trans_func()
        for name, func, col, targetCol, preTransform in self.transFunctions_:
            # if BaseCfg.debug:
            # logger.debug(f'fit {name}:{col}')
            if hasattr(func, 'fit'):
                if col:
                    if col in X.columns:  # col may not generated by previous transformers
                        func.fit(X[col])
                    else:
                        logger.warning(f'{col} is not in X columns yet')
                else:
                    func.fit(X)
            if preTransform:  # some transformers need to be applied before others to fit the data
                X = _transform_single(func, col, targetCol, X)
        return self

    def transform(self, X):
        # add columns for targetCol if not exist
        cols = self.get_feature_names_out()
        to_add_cols = [col for col in cols if col not in X.columns]
        X = addColumns(X, to_add_cols, float('nan'))
        num_procs = min((psutil.cpu_count(logical=False) - 1),
                        CONCURRENT_PROCESSES_MAX,
                        self.num_procs,
                        len(X.index),
                        (getMemoryLimitedExtraProcessNumber()+1))
        if (num_procs == 1) or (X.shape[0] < MULTI_PROCESS_THRESHOLD):
            return _transform(simpleTransformer=self, X=X)
        return self.concurrent_transform(X, num_procs)

    def concurrent_transform(self, X, num_procs):
        logger.info(f'num_procs: {num_procs}')
        timer = Timer(f'transform', logger)
        df_results = []
        splitted_df = np.array_split(X, num_procs)
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_procs) as executor:
            # print('before submit========================')
            # print(splitted_df)
            results = [
                executor.submit(_transform, simpleTransformer=self, X=df)
                for df in splitted_df
            ]
            # print('after submit========================')
            # print(results)
            for result in concurrent.futures.as_completed(results):
                try:
                    df_results.append(result.result())
                except Exception as ex:
                    logger.error(ex)
                    raise ex
        df_results = pd.concat(df_results)
        print('after merge========================\n')
        timer.stop(len(X.index))
        return df_results


def _transform(simpleTransformer, X):
    for name, func, col, targetCol, preTransform in simpleTransformer.transFunctions_:
        timer = Timer(f'transform:{name}', logger)
        # logger.debug(f'transform {name}:{col}=>{targetCol}')
        X = _transform_single(func, col, targetCol, X)
        timer.stop(len(X.index))
    #X.to_excel("sh.xlsx")
    return X


def _transform_single(func, col, targetCol, X):
    if hasattr(func, 'transform'):  # transformer
        if col:  # column to column
            if col in X.columns:
                X[targetCol] = func.transform(X[col])
            else:
                logger.error(f'Column {col} not found')
        else:  # DataFrame to DataFrame
            X = func.transform(X)
    else:  # function
        if col:  # column to column
            if col in X.columns:
                result = X.apply(
                    lambda x: func(x, x[col]), axis=1)
                if isinstance(targetCol, list):
                    logger.warning(
                        f'targetCol is list: {col} => {targetCol}')
                    result = pd.DataFrame(list(result))
                X[targetCol] = result
            else:
                logger.error(f'Column {col} not found')
        else:  # DataFrame to DataFrame
            X = X.apply(lambda x: func(x), axis=1)
    return X
