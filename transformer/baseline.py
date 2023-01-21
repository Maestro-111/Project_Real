
from datetime import datetime
from math import isnan
import re
import pandas as pd
import numpy as np
from base.base_cfg import BaseCfg
from base.mongo import getMongoClient
from base.timer import Timer
from base.util import dateFromNum
from sklearn.base import BaseEstimator, TransformerMixin
from transformer.const_label_map import getLevel

logger = BaseCfg.getLogger(__name__)
FEATURES_KEY = 'features'
VALUES_KEY = 'values'
RATIOS_KEY = 'ratios'
SALETP_COL = 'saletp-b'

EXT_N = '-bl-n'
EXT_M = '-blm-n'


def findColumnName(df, col):
    for ext in ['-n', '-b', '-l', '-e', '']:
        if f"{col}{ext}" in df.columns:
            return f"{col}{ext}"
    for colInDf in df.columns:
        if colInDf.startswith(col) and col.endswith('-n'):
            return colInDf
    return None


class BaselineTransformer(BaseEstimator, TransformerMixin):
    """baseline transformer is a statistics collector and transformer.
    transform baseline features to dollar measurable numeric values.

    Parameters
    ----------
    None
    """
    discrete_cols = ['cmty', 'gr', 'tgr', 'bdrms',
                     'br_plus', 'bthrms', 'kch', 'onD-month', ]
    # continous columns are also for sale properties only. rent properties do not have these columns
    continous_cols = ['flt', 'depth', 'tax', 'mfee', 'bltYr', 'sqft', ]

    def __init__(
        self,
        sale: bool = None,
        collection: str = None,
    ):
        self.sale = sale  # True for sale, False for rent, None for both
        # when collection is not None, save the stats to the collection
        self.collection = collection
        pass

    def get_feature_names_out(self):
        self.discrete_new_cols = [f"{col}{EXT_N}" for col in self.discrete_cols] + \
            [f"{col}{EXT_M}" for col in self.discrete_cols]
        # for rent
        if self.sale == False:
            return self.discrete_new_cols

        # for sale
        self.continous_new_cols = [f"{col}{EXT_N}" for col in self.continous_cols] + \
            [f"{col}{EXT_M}" for col in self.continous_cols]
        return self.discrete_new_cols + self.continous_new_cols

    def _connect_db(self):
        if hasattr(self, '_db_connected') and self._db_connected:
            return self._db_connected
        if not self.collection:
            logger.warn("No collection specified")
            return
        # create index on col
        MongoDB = getMongoClient()
        if not MongoDB.hasIndex(self.collection, 'col'):
            MongoDB.createIndex(self.collection, fields=[
                ("col", 1), ('i', 1)], unique=True)
        self._db_connected = MongoDB
        return MongoDB

    def fit(self, X, y=None):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : dataframe
            The training input samples.
        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        self : object
            Returns self.
        """
        timer = Timer('baseline', logger)
        timer.start()
        totalCount = 0
        self.stats_ = [{}, {}]
        SP_N = 'sp-n'
        # X = X.where(X[SP_N] > 0).dropna()
        XX = [None, None]
        XX[0] = X.loc[(X[SALETP_COL] == 0) & (X[SP_N] > 0)]
        XX[1] = X.loc[(X[SALETP_COL] == 1) & (X[SP_N] > 0)]

        for i in [0, 1]:
            self.stats(self.stats_[i], XX[i], SP_N)
        if self.collection is not None:
            self.saveStats()
        timer.stop(totalCount)
        return self

    def saveStats(self):
        totalRecords = 0
        if self.collection is None:
            logger.warning('No collection to save to')
            return totalRecords
        mongo = getMongoClient()
        for i in range(2):
            stats = self.stats_[i]
            YYMM = stats['saveKey']
            saletp = 's' if i == 0 else 'r'
            keyPrefix = f"{YYMM}{saletp}"
            record = {
                '_id': keyPrefix,
                'ts': datetime.now(),
                'count': stats['count'],
                'seconds': stats['seconds'],
                'dateSpan': stats['dateSpan'],
                'startOnD': stats['startOnD'],
                'endOnD': stats['endOnD'],
            }
            totalRecords += 1
            mongo.save(self.collection, record)
            for key in stats:
                if isinstance(key, tuple):
                    cityTypeStats = stats[key]
                    record = {
                        '_id': f"{keyPrefix}:{key[0]}:{key[1]}:{key[2]}",
                        'count': cityTypeStats['count'],
                        'mean': cityTypeStats['mean'],
                        'std': cityTypeStats['std'],
                        'min': cityTypeStats['min'],
                        '25%': cityTypeStats['25%'],
                        '50%': cityTypeStats['50%'],
                        '75%': cityTypeStats['75%'],
                        'max': cityTypeStats['max'],
                    }
                    totalRecords += 1
                    mongo.save(self.collection, record)
                    for feature in cityTypeStats[FEATURES_KEY]:
                        featureStats = cityTypeStats[FEATURES_KEY][feature]
                        record = {
                            '_id': f"{keyPrefix}:{key[0]}:{key[1]}:{key[2]}:{feature}"}
                        if isinstance(featureStats.get(VALUES_KEY, None), dict):
                            record[VALUES_KEY] = {}
                            # logger.debug(
                            #     f"featureStats[VALUES_KEY]: {featureStats[VALUES_KEY]}")
                            for value in featureStats[VALUES_KEY]:
                                record[VALUES_KEY][f'{value}'] = featureStats[VALUES_KEY][value]
                            for fkey in featureStats:
                                if fkey != VALUES_KEY:
                                    record[fkey] = featureStats[fkey]
                        else:
                            record = record | featureStats
                        totalRecords += 1
                        try:
                            mongo.save(self.collection, record)
                        except Exception as e:
                            logger.error(e)
                            logger.error(record)
                            logger.error(featureStats.get(
                                VALUES_KEY, 'Not found'))
                            logger.error(
                                type(featureStats.get(VALUES_KEY, None)))

        return totalRecords

    def stats(self, stats_, X, SP_N):
        stats_['count'] = X.shape[0]
        startTs = datetime.now()
        dfOverall = X.groupby(
            ['prov', 'city', 'ptype2-l'])[SP_N].describe()
        logger.info(
            f"do stats: {[col for col in X.columns if 'sp' in col]}")
        logger.info(f"do stats: {dfOverall.head()}")
        logger.info(X[['sp', SP_N]].head(100))
        for index, row in dfOverall.iterrows():
            key = (index[0], index[1], index[2])  # prov-city-ptype2-l
            dict = row.to_dict()
            dict[FEATURES_KEY] = {}
            stats_[key] = dict
            logger.info(f"stats: {key} {dict}")
        for col in self.discrete_cols:
            col_n = self.featureOverallStats(stats_, SP_N, X, col)
            if col_n is None:
                continue
            self.discreteStats(stats_, X, SP_N, col, col_n)
        if self.sale == True or self.sale is None:  # rent properties do not have continous columns
            for col in self.continous_cols:
                col_n = self.featureOverallStats(stats_, SP_N, X, col)
                if col_n is None:
                    continue
                self.continousStats(stats_, X, SP_N, col, col_n)
        onDStats = X['onD'].describe()
        endOnD = dateFromNum(onDStats['max'])
        startOnD = dateFromNum(onDStats['min'])
        dateSpan = (endOnD - startOnD).days
        stats_['dateSpan'] = dateSpan
        stats_['startOnD'] = startOnD.strftime('%Y-%m-%d')
        stats_['endOnD'] = endOnD.strftime('%Y-%m-%d')
        stats_['saveKey'] = endOnD.strftime('%Y%m')
        endTs = datetime.now()
        seconds = (endTs - startTs).total_seconds()
        stats_['seconds'] = seconds

    def featureOverallStats(self, stats_, SP_N, X, col):
        col_n = findColumnName(X, col)
        if col_n is None:
            logger.warning(f'column {col} not found')
            return col_n
        df = X.groupby(
            ['prov', 'city', 'ptype2-l'])[col_n].describe()
        for index, row in df.iterrows():
            # prov-city-ptype2-l
            key = (index[0], index[1], index[2])
            feature_stats = row.to_dict()
            feature_stats['col'] = col_n
            # stats for each feature
            stats_[key][FEATURES_KEY][col] = feature_stats
            # logger.info(f"{col} {key} {feature_stats}")
        return col_n

    def discreteStats(self, stats_, X, SP_N, col, col_n):
        df = X.groupby(
            ['prov', 'city', 'ptype2-l', col_n])[SP_N].describe()
        for index, row in df.iterrows():
            # prov-city-ptype2-l
            key = (index[0], index[1], index[2])
            stats = stats_[key]
            feature_stats = stats[FEATURES_KEY][col]
            feature_value_stats = row.to_dict()
            if VALUES_KEY not in feature_stats:
                feature_stats[VALUES_KEY] = {}
            # delta of value to mean
            feature_mean = feature_stats.get('mean', stats.get('mean', 0))
            feature_median = feature_stats.get('50%', stats.get('50%', 0))
            feature_value_mean = feature_value_stats['mean']
            feature_value_median = feature_value_stats['50%']
            delta = feature_value_mean - feature_mean
            # delta of value to median
            deltaM = feature_value_median - feature_median
            # debug
            logger.info(
                f"discreteStats {col} {key} {feature_mean} {feature_median} {feature_stats}")
            # price/delta ratio by mean
            if delta == 0:
                ratio = 0
            else:
                ratio = (
                    feature_value_mean - stats['mean']) / delta
            # price/delta ratio by median
            if deltaM == 0:
                ratioM = 0
            else:
                ratioM = (
                    feature_value_median - stats['50%']) / deltaM
            feature_value_stats['delta'] = delta
            feature_value_stats['deltaM'] = deltaM
            feature_value_stats['ratio'] = ratio
            feature_value_stats['ratioM'] = ratioM
            feature_stats[VALUES_KEY][index[3]] = feature_value_stats

    def continousStats(self, stats_, X, SP_N, col, col_n):
        # calculate delta ratio
        X = X.copy()
        col_n_delta = f"{col}-delta"
        col_n_sp_diff = f"{col}-sp-diff"
        col_n_ratio = f"{col}-ratio"
        X[[col_n_delta, col_n_sp_diff, col_n_ratio]] = 0

        def _calcRatio(row):
            _stats = stats_.get(
                (row['prov'], row['city'], row['ptype2-l']), None)
            if _stats is None:
                logger.warning(
                    f"stats not found for {row['prov']} {row['city']} {row['ptype2-l']}")
                return row
            # 1. calculate value delta: delta = value - mean
            delta = row[col_n] - _stats[FEATURES_KEY][col]['mean']
            row[col_n_delta] = delta
            # 2. calculate soldprice difference
            sp_diff = row[SP_N] - _stats['mean']
            row[col_n_sp_diff] = sp_diff
            # 3. calculate ratio
            if delta != 0:
                row[col_n_ratio] = sp_diff / delta
            logger.info(
                f"ratio {col} {row[col_n_ratio]} {sp_diff} / {delta}({row[col_n]} - {_stats[FEATURES_KEY][col]['mean']})")
            return row
        X = X.apply(_calcRatio, axis=1, result_type='broadcast')
        # calculate delta ratio stats
        df = X.groupby(
            ['prov', 'city', 'ptype2-l'])[col_n_ratio].describe(include=np.number)
        for index, row in df.iterrows():
            # prov-city-ptype2-l
            key = (index[0], index[1], index[2])
            feature_stats = row.to_dict()
            # stats for each feature
            stats_[key][FEATURES_KEY][col][RATIOS_KEY] = feature_stats

    def transform(self, X):
        """Transform X.

        Parameters
        ----------
        X : dataframe. The input samples.

        Returns
        -------
        X_transformed : dataframe. The transformed data.
        """
        # logger.debug(f'transform rms')
        timer = Timer('baseline', logger)
        timer.start()
        totalCount = 0
        transformedCount = 0
        new_cols = self.get_feature_names_out()
        X[new_cols] = 0

        def _getFeatureStats(row, col, stats):
            featureStats = stats[FEATURES_KEY][col]
            if featureStats is None:
                return None, None
            col_n = featureStats['col']
            if col_n is None:
                logger.warning(f'column {col} not found in featureStats')
                return None, None
            return featureStats, col_n

        def _transformDiscrete(row, col, stats):
            featureStats, col_n = _getFeatureStats(row, col, stats)
            if featureStats is None:
                return
            # set feature value
            valueStats = featureStats[VALUES_KEY].get(row[col_n], None)
            if valueStats is None:
                return
            if row[col+EXT_N] != 0:
                logger.warning(
                    f'column {col} has value {row[col+EXT_N]} already')
            row[col+EXT_N] = valueStats['mean']
            row[col+EXT_M] = valueStats['50%']

        def _transformContinuous(row, col, stats):
            featureStats, col_n = _getFeatureStats(row, col, stats)
            if featureStats is None:
                return
            # set feature value
            ratioStats = featureStats.get(RATIOS_KEY, None)
            if ratioStats is None:
                return
            row[col+EXT_N] = ratioStats['mean'] * row[col_n] + stats['mean']
            row[col+EXT_M] = ratioStats['50%'] * row[col_n] + stats['mean']

        def _transform(row):
            nonlocal totalCount, transformedCount
            totalCount += 1
            key = (row['prov'], row['city'], row['ptype2-l'])
            saletp = row[SALETP_COL]
            if saletp not in [0, 1]:
                logger.warning(f'invalid saletp {saletp}')
                return row
            stats = self.stats_[saletp].get(key, None)
            if stats is None:
                return row
            for col in self.discrete_cols:
                _transformDiscrete(row, col, stats)
            if saletp == 1:
                return row
            for col in self.continous_cols:
                _transformContinuous(row, col, stats)
            transformedCount += 1
            return row
        X = X.apply(_transform, axis=1, result_type='broadcast')
        timer.stop(totalCount)
        logger.info(f'transformed {transformedCount}/{totalCount}')
        return X
