
from base.base_cfg import BaseCfg
from base.const import CONCURRENT_PROCESSES_MAX
from base.timer import Timer
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import psutil
import numpy as np
import concurrent.futures

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
        self.num_procs = num_procs
        self.transFunctions = []
        self.transFunctionsByName = {}
        for trans in transFunctions:
            if len(trans) == 2:
                self.transFunctions.append((*trans, None, None))
            elif len(trans) == 3:
                self.transFunctions.append((*trans, trans[2]))
            elif len(trans) == 4:
                self.transFunctions.append(trans)
            else:
                raise ValueError('Invalid number of parameters in list(tuple)')
            self.transFunctionsByName[trans[0]] = trans

    def fit(self, X, y=None):
        for name, func, col, targetCol in self.transFunctions:
            if BaseCfg.debug:
                logger.debug(f'fit {name}:{col}')
            if getattr(func, 'fit', None) is not None:
                if col:
                    if col in X.columns:  # col may not generated by previous transformers
                        func.fit(X[col])
                    else:
                        logger.info(f'{col} is not in X columns yet')
                else:
                    func.fit(X)
        return self

    def transform(self, X):
        if self.num_procs == 1:
            return _transform(simpleTransformer=self, X=X)
        return self.concurrent_transform(X)

    def concurrent_transform(self, X):
        num_procs = min((psutil.cpu_count(logical=False) - 1),
                        CONCURRENT_PROCESSES_MAX,
                        self.num_procs,
                        len(X.index))
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
                    logger.error(str(ex))
                    raise ex
        df_results = pd.concat(df_results)
        # print('after merge========================')
        # print(df_results)
        timer.stop(len(X.index))
        return df_results


def _transform(simpleTransformer, X):
    for name, func, col, targetCol in simpleTransformer.transFunctions:
        timer = Timer(f'transform:{name}', logger)
        logger.debug(f'transform {name}:{col}=>{targetCol}')
        if getattr(func, 'transform', None) is not None:
            if col:
                X.loc[:, targetCol] = func.transform(X[col])
            else:
                X = func.transform(X)
        else:
            if col:
                if col in X.columns:
                    result = X.apply(
                        lambda x: func(x, x[col]), axis=1)
                    if isinstance(targetCol, list):
                        result = pd.DataFrame(list(result))
                    X.loc[:, targetCol] = result
                else:
                    logger.warning('Column {col} not found')
            else:
                X = X.apply(lambda x: func(x), axis=1)
            pass
        # print(f'after {name} ------------------------------')
        # print(X)
        timer.stop(len(X.index))
    return X
