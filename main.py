import os
os.environ['DEBUG'] = 'True'
os.environ["RMBASE_FILE_PYTHON"] = "/Users/fred/work/cfglocal/built/base.py"
print(os.environ['RMBASE_FILE_PYTHON'])

from data.data_source import read_data_by_query
import numpy as np
import pandas as pd
import logging
from base.const import Mode
from helper.feature_select import FeatureSelector
from base.timer import Timer
from estimator.organizer import Organizer
from math import isnan


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logging.debug("test")


org = Organizer()
org.load_data()
org.init_transformers()

org.train_models()

org.save_models()

df, a_cols = org.predict([
    'TRBW5841870', 'TRBW5841549', 'TRBW5840562', 'DDF25100869', 'TRBX5810806',  # Mississauga
    'TRBC5841835', 'TRBC5841873', 'TRBE5841152', 'TRBE5840727', 'TRBC5841124',  # Toronto
    'TRBN5841327', 'DDF25052539', 'TRBN5841525', 'TRBN5840799', 'TRBN5841438',  # Markham
    'TRBW5841599', 'TRBW5841869', 'TRBW5841824', 'TRBW5840755', 'TRBW5840758',  # Oakville
    'TRBW5840555', 'TRBW5841859', 'TRBW5841673', 'TRB40323381', 'TRB40297792',  # Brampton
])
a_cols.append('lp-n')
df[a_cols]


df['onD-week-non', 'D-season-n']
for col in a_cols:
    if col in df.columns:
        print(df[col])
    else:
        print(f'{col} not found')

# ----------------------------------------------------------
# start testing below

builtYear = BuiltYearLgbmManager(org.data_source)
builtYear.train()


sqft = SqftLgbmEstimateManager(org.data_source)
sqft.train()


sold = SoldPriceLgbmEstimateManager(org.data_source)
sold.train()

# -------------------
ds = org.data_source
df = ds.load_df_grouped(['TRBW5841870'], org.root_preprocessor)
ds.prov_city_to_area = {('ON', 'Mississauga'): 'Peel'}

df['_id']

df = read_data_by_query({'_id': {'$in': ['TRBW5841870']}}, ds.col_list)
print('prov' in df.columns)
print(df['prov'])

df.apply(lambda row: print(row['city']), axis=1)
df.apply(lambda row: print((row['prov'], row['city'])), axis=1)
(df.loc[0, 'prov'], df.loc[0, 'city'])
print(ds.prov_city_to_area[(df['prov'], df['city'])])

trans = org.root_preprocessor.customTransformer.transFunctionsByName_[
    'fce_x'][1]
trans.fit(ds.df_raw)
df_raw = ds.df_raw
Xs = df_raw['fce'].apply(lambda x: trans.map_.get(x, x))
Xs1 = Xs.explode(ignore_index=False)
Xs1

col_labels = Xs1.value_counts()
col_labels.index = col_labels.index.astype(str)
col_labels = col_labels.sort_values(ascending=False)


# test built year
def needBuiltYear(row):
    return not (row['bltYr-n'] is None or isnan(row['bltYr-n']))


df = em.load_data(
    scale=org.default_all_scale,
    x_columns=[
        'lat', 'lng',
        'gatp', 'zip',
        'st_num-st-n',
        'bthrms', 'pstyl', 'ptp',
        'bsmt',  'heat', 'park_fac',
        'rms', 'bths',
        'st', 'st_num',
    ],
    y_column='bltYr',
    date_span=180,
    filter_func=needBuiltYear,
    suffix_list=['-n', '-c', '-b'],
)


printColums(df)


# ----------------------------------------------------------
# other stuff

def printColums(df, start: str = None):
    cols = []
    for c in df.columns:
        if start is not None:
            if c.startswith(start):
                cols.append(c)
        else:
            cols.append(c)
    cols.sort()
    for c in cols:
        print(c)


class A:
    a = 10
    b = [1, 2]

    def __init__(self):
        self.c = [4, 5]


a = A()

a.b


print(hasattr(a, 'b'))

df1 = pd.DataFrame([[1, 2, 0], [4, 5, 0], [7, 8, 0], [7, 0, 0]],
                   index=['cobra', 'viper', 'sidewinder', 'zero'],
                   columns=['max_speed', 'shield', 'zzz'])
labels = df1['max_speed'].value_counts()
labels.index = labels.index.astype(str)
labels = labels.sort_index()

for i, c in labels.items():
    print(f'{i}=>{c}')

tuples = [
    ('cobra', 'mark i'), ('cobra', 'mark ii'),
    ('sidewinder', 'mark i'), ('sidewinder', 'mark ii'),
    ('viper', 'mark ii'), ('viper', 'mark iii')
]
index = pd.MultiIndex.from_tuples(tuples)
values = [[12, 2], [0, 4], [10, 20],
          [1, 4], [7, 1], [16, 36]]
df1 = pd.DataFrame(values, columns=['max_speed', 'shield'], index=index)

df2 = df1.loc[:, (df1 != 0).any(axis=0)]
df2

org.train_writeback_predictors()


print(org.root_preprocessor.get_feature_columns())


for i in [0, 1, 2]:
    pred = org.writeback_predictors[i]
    pred.feature_select()

pred = org.writeback_predictors[2]
print(pred.df.shape)
print(pred.prefer_estimated_value)
print(pred.xumeric_columns_)
df = pred.df
df = saleDF
cols = df.columns
for col in cols:
    naCnt = df[col].isna().sum()
    print(f'{col}:{naCnt}')


# features selection
saleDF = org.data_source.df_grouped.loc[(0), :]
saleDF.drop(['lp', 'lp-n', 'lpr', 'lpr-n', 'tax-n',
            'taxyr-n'], axis='columns', inplace=True)
myFs = FeatureSelector(saleDF)
name_scores_bltYr, scores = myFs.find_features(
    'bltYr-n', exclude_cols=['bltYr-e'])
myFs.plot_scores(scores)
print(scores)


[l1, l2, l3, af] = myFs.find_raw_features('sqft-n', exclude_cols=['sqft-e'])
print(l1)
print(l2)
print(l3)
print(af)

scores = [x[1] for x in name_scores]


name_scores_sqft, scores = myFs.find_features(
    'sqft-n', exclude_cols=['sqft-e'])
myFs.plot_scores(scores)
name_scores_sp, scores = myFs.find_features('sp-n', exclude_cols=['value-e'])
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


printColums(data_source.df_grouped)


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

str(20120312)


df = pd.DataFrame({'points': [14, 19, 9, 21, 25, 29, 20, 11],
                   'assists': [5, 7, 7, 9, 12, 9, 9, 4],
                   'rebounds': [11, 8, 10, 6, 6, 5, 9, 12]})
df.mean(axis=0)