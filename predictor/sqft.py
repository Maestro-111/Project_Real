import copy
import datetime
from math import isnan


import pandas as pd
from predictor.base_predictor import BasePredictor
from prop.data_source import DataSource

from base.base_cfg import BaseCfg
from base.util import print_dateframe
from prop.estimate_scale import EstimateScale, PropertyType
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import Normalizer

import lightgbm as lgb

logger = BaseCfg.getLogger(__name__)


def needSqft(row):
    return not (row['sqft_n'] is None or isnan(row['sqft_n']))


class Sqft(BasePredictor):
    """Sqft Estimator class"""
    base_model_params = {
        'n_estimators': 300,
        'max_depth': -1,
        'num_leaves': 100,
    }

    def __init__(
        self,
        data_source: DataSource,
        scale: EstimateScale = None,
        model_store=None,
    ):
        super().__init__(
            name='Sqft',
            data_source=data_source,
            source_filter_func=needSqft,
            source_date_span=365*18,
            source_suffix_list=['_n', '_c'],  # , '_b'],
            scale=scale,
            model_store=model_store,
            y_numeric_column='sqft_n',
            y_column='sqft_n',
            x_columns=[
                'lat', 'lng',
                'zip',  'gatp',
                'st', 'st_num',
                'bthrms', 'pstyl',
                'ptp', 'pstyl', 'constr', 'feat',
                'bsmt',  'heat', 'park_fac',
                'depth', 'flt', 'rms', 'bths',
                'sqft',
            ],
        )
        # self.col_list = [
        #     'lat', 'lng', 'rmBltYr',
        #     'cs16', 'st_num', 'st', 'zip', 'sid',
        #     'gatp', 'flt', 'depth', 'gr', 'tgr', 'pstyl', 'bthrms',
        # ]
        # self.y_column = 'rmBltYr'
        # self.x_columns = [
        #     'lat', 'lng',
        #     'zip',  'gatp', 'bthrms',
        #     'st', 'st_num', 'st_num_st',
        # ]
        # self.categorical_feature = ['st', 'zip', 'gatp']

    def prepare_model(self):
        super().prepare_model()
        if self.model is None:
            self.model_params = copy.copy(Sqft.base_model_params)
            self.model = lgb.LGBMRegressor(**self.model_params)
            logger.info('Sqft: model_params: {}'.format(
                self.model_params))

    def prepare_data(self, X, params=None):
        """Prepare data for training"""
        X = X.copy()
        # poly = PolynomialFeatures(2)
        # poly.fit_transform(X_train)
        X['st_c'].fillna(0, inplace=True)
        X['st_num_st_n'] = (X['st_c'].astype(int) + 1) * 100000 + X['st_num_n']
        X.dropna(inplace=True)
        self.generate_numeric_columns()
        return X


def testSqftByType(propType: PropertyType, city: str = 'Toronto'):
    scale = EstimateScale(
        datePoint=datetime.datetime(2022, 2, 1, 0, 0),
        propType=propType,
        prov='ON',
        city=city,
    )
    builtYearEstimator = Sqft()
    builtYearEstimator.set_scale(scale)
    builtYearEstimator.set_query(query={'rmBltYr': {'$ne': None}})
    # best_span, best_score = builtYearEstimator.tune()
    # print('**********************')
    # print(f'best: {best_score} => {best_span} days',
    #       scale.propType, scale.city)
    # print('**********************')
    builtYearEstimator.load_data(
        date_span=2922, query=builtYearEstimator.db_query)
    score = builtYearEstimator.train()
    print('**********************')
    print(f'{scale.city}:{propType} => {score}')
    print('**********************')


def testSqft():
    # for propType in [PropertyType.CONDO, PropertyType.TOWNHOUSE, PropertyType.SEMI_DETACHED, PropertyType.DETACHED]:
    #     for city in ['Toronto', 'Mississauga', 'Brampton', 'Markham', 'Oakville']:
    for propType in [PropertyType.DETACHED]:
        for city in ['Toronto', 'Mississauga', 'Brampton', 'Markham', 'Oakville']:
            testSqftByType(propType, city)
        print('----------------------------------------------------------------')


if __name__ == '__main__':
    testSqft()
