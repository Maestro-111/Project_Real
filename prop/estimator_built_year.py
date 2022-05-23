import datetime
from numpy import mean, std

import pandas as pd
from sklearn.model_selection import RepeatedKFold, cross_val_score
from base.base_cfg import BaseCfg
from base.util import print_dateframe
from prop.estimate_scale import EstimateScale, PropertyType
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import Normalizer

import lightgbm as lgb


class BuiltYear():
    """Built Year Estimator class"""
    name: str = 'built_year'

    def __init__(self):
        self.col_list = ['lat', 'lng', 'rmBltYr',
                         'cs16', 'st_num', 'st', 'zip', 'sid',
                         'gatp', 'flt', 'depth', 'gr', 'tgr', 'pstyl', 'bthrms']
        self.y_column = 'rmBltYr'
        self.x_columns = ['lat', 'lng',
                          'zip',  'gatp', 'bthrms',
                          'st', 'st_num', 'st_num_st']
        self.categorical_feature = ['st', 'zip', 'gatp']
        self.model.model_param = {
            'n_estimators': 300,
            'max_depth': -1,
            'num_leaves': 100}

    def _train(self, X_train, y_train):
        """Train the model"""
        # poly = PolynomialFeatures(2)
        # poly.fit_transform(X_train)

        if BaseCfg.isDebug():
            print_dateframe(X_train)
        # single training
        self.model.train(X_train, y_train)
        # cross validation with k-fold
        cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
        n_scores = cross_val_score(
            self.model.model, X_train, y_train, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')
        print('MAE: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))

    def prepare_data(self, X, params=None):

        # X['st_num'] = pd.to_numeric(X['st_num'], errors='coerce')
        # X = X.dropna(subset=['st_num'])
        X = self.model.str_to_num(X, ['st_num', 'bthrms'])

        # set pstyl default
        # X['pstyl'] = X['pstyl'].fillna('2-Storey')
        # print_dateframe(X)
        # X = X.drop(X.loc[X['st'].apply(lambda x: type(x) != str)].index)
        # X = X.drop(X.loc[X['zip'].apply(lambda x: type(x) != str)].index)
        X = self.model.drop_columns_not_in_type(
            X, self.categorical_feature, str)

        X = self.model.encode_label(X, self.categorical_feature)

        X['st_num_st'] = (X['st'].astype(int) + 1) * 100000 + X['st_num']
        # X['st_num_st'] = Normalizer().fit(
        #     X['st_num_st']).transform(X['st_num_st'])

        return X


def testBuiltYearByType(propType: PropertyType, city: str = 'Toronto'):
    scale = EstimateScale(
        datePoint=datetime.datetime(2022, 2, 1, 0, 0),
        propType=propType,
        prov='ON',
        city=city,
    )
    builtYearEstimator = BuiltYear()
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


def testBuiltYear():
    # for propType in [PropertyType.CONDO, PropertyType.TOWNHOUSE, PropertyType.SEMI_DETACHED, PropertyType.DETACHED]:
    #     for city in ['Toronto', 'Mississauga', 'Brampton', 'Markham', 'Oakville']:
    for propType in [PropertyType.DETACHED]:
        for city in ['Toronto', 'Mississauga', 'Brampton', 'Markham', 'Oakville']:
            testBuiltYearByType(propType, city)
        print('----------------------------------------------------------------')


if __name__ == '__main__':
    testBuiltYear()
