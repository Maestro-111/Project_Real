import logging
import pandas as pd
import numpy as np

import os
os.environ['DEBUG'] = 'True'
os.environ["RMBASE_FILE_PYTHON"] = "/Users/fred/work/cfglocal/built/base.py"
print(os.environ['RMBASE_FILE_PYTHON'])
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logging.debug("test")

from base.const import Mode
from helper.feature_select import FeatureSelector
from base.timer import Timer
from organizer import Organizer



org = Organizer()
org.init_transformers()

org.train_writeback_predictors()

for i in [0, 1, 2]:
    pred = org.writeback_predictors[i]
    pred.feature_select()

pred = org.writeback_predictors[2]
print(pred.df.shape)
print(pred.prefer_estimated_value)
print(pred.x_numeric_columns_)
df = pred.df
cols = df.columns
for col in cols:
    naCnt = df[col].isna().sum()
    print(f'{col}:{naCnt}')


# features selection
saleDF = org.data_source.df_grouped.loc[(0), :]
saleDF.drop(['lp', 'lp-n', 'lpr', 'lpr-n', 'tax-n',
            'taxyr-n'], axis='columns', inplace=True)
myFs = FeatureSelector(saleDF)
name_scores_bltYr, scores = myFs.find_features('bltYr-n')
myFs.plot_scores(scores)
name_scores_sqft, scores = myFs.find_features('sqft-n')
myFs.plot_scores(scores)
name_scores_sp, scores = myFs.find_features('sp-n')
myFs.plot_scores(scores)


rentDF = org.data_source.df_grouped.loc[(slice(1, 1), slice(
    None), slice(None), slice(None), slice(None)), :]
rentDF.drop(['lp', 'lp-n', 'tax-n', 'taxyr-n'], axis='columns', inplace=True)
rentFs = FeatureSelector(rentDF)
name_scores_rent, scores = rentFs.find_features('sp-n')
rentFs.plot_scores(scores)

rentDF.head(5).to_csv('rent.csv')


pred.df.loc[slice(0, 100), ['sqft-n', 'sqft-e',
                            'bltYr-n', 'bltYr-e']].to_csv('df4.csv')


data_source = org.data_source
data_source.df_grouped = data_source.df_grouped.drop(
    labels=['bltYr-e', 'sqft-e', 'value-e', 'rent_value-e'], axis='columns')
print(data_source.df_grouped.columns)


org.train_predictors()

org.data_source.load_data(org.root_preprocessor)
org.root_preprocessor.


def printColums(df, start: str = None):
    for c in df.columns:
        if start is not None:
            if c.startswith(start):
                print(c)
        else:
            print(c)


printColums(data_source.df_grouped, 'val')

data_source.df_grouped.to_csv('data3.csv')


slices = [slice(0, 0, None), slice('Detached', 'Detached', None), slice('ON', 'ON', None), slice(
    'Toronto', 'Toronto', None), slice('Toronto', 'Toronto', None), slice(None, None, None)]
rd = data_source.df_grouped.loc[tuple(slices), :]
print(rd.shape)


pred = org.predictors[0]


df_test_y = pred.predict(pred.df_test[pred.x_numeric_columns_])

print(len(df_test_y))

print(pred.df_test.loc[1, :])
print(pred.df)


class C:
    def __init__(self, mode: Mode = Mode.TRAIN):
        self.mode = mode

    def show(self):
        print(f'{self.mode} {self.mode == Mode.TRAIN}')
        print(self.mode == Mode.TRAIN)
        print(self.mode is Mode.TRAIN)


c1 = C()
c1.show()
c2 = C(Mode.PREDICT)
c2.show()

pred.df = None
pred.df_prepared = None
pred.load_data()
pred.train()
pred.cross_val()

l = ['a-e', 'a-n']
m = l.copy()
for i in l:
    if i.endswith('-n'):
        m.remove(i)
print(m, l)


# simple steps
org = Organizer()
org.init_transformers()
org.train_predictors()