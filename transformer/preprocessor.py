
import datetime
from gc import garbage
from itertools import chain

import pandas as pd
import re
from math import isnan
from pyrsistent import v

from base.util import allTypeToFloat, allTypeToInt, stringToInt

from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from base.base_cfg import BaseCfg
from base.const import NONE, RENT_PRICE_UPPER_LIMIT, SALE_PRICE_LOWER_LIMIT, UNKNOWN, DROP, MEAN, Mode
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from data.estimate_scale import PropertyType, PropertyTypeRegexp
from transformer.binary import BinaryTransformer
from transformer.baths import BthsTransformer
from transformer.bedrooms import RmsTransformer
from transformer.dates import DatesTransformer
from transformer.db_one_hot_array import DbOneHotArrayEncodingTransformer
from transformer.select_col import SelectColumnTransformer
from transformer.db_label import DbLabelTransformer
from transformer.db_numeric import DbNumericTransformer
from transformer.drop_row import DropRowTransformer
from transformer.const_label_map import getLevel, levelType, acType, \
    bsmtType, featType, constrType, garageType, lockerType, \
    heatType, fuelType, exposureType, laundryType, \
    parkingDesignationType, parkingFacilityType, balconyType, \
    ptpType
from transformer.simple_column import SimpleColumnTransformer
from transformer.street_n_st_num import StNumStTransformer

logger = BaseCfg.getLogger(__name__)


def yearOfDateNumber(dateNumber, deltaDays=0):
    date = datetime.datetime.strptime(str(dateNumber), '%Y%m%d')
    date = date + datetime.timedelta(days=deltaDays)
    return date.year


def yearOfByField(row, field, deltaDays=0):
    return yearOfDateNumber(row[field], deltaDays)


def yearOf(field, deltaDays=0):
    return lambda row: yearOfByField(row, field, deltaDays)


def saletpSingleValue(row, saletp):
    if isinstance(saletp, str):
        return saletp
    elif isinstance(saletp, tuple) or isinstance(saletp, list):
        if len(saletp) == 1:
            return saletp[0]
        elif len(saletp) == 0:
            return 'Sale'
        if row['lpr'] is None or row['lpr'] == 0:
            return 'Sale'
        elif row['lp'] is None or row['lp'] == 0:
            return 'Lease'
    return None


def binarySaletpByRow(row, saletp):
    value = saletpSingleValue(row, saletp)
    if value == 'Sale':
        return 0
    elif value == 'Lease':
        return 1
    elif value is not None:
        logger.error('Unknown saletp value', value, row['_id'])
    return None


def ptype2SingleValue(row, ptype2):
    for value in ptype2:
        if not isinstance(value, str):
            continue
        if PropertyTypeRegexp.SEMI_DETACHED.match(value):
            return PropertyType.SEMI_DETACHED
        elif PropertyTypeRegexp.DETACHED.match(value):
            return PropertyType.DETACHED
        elif PropertyTypeRegexp.TOWNHOUSE.match(value):
            return PropertyType.TOWNHOUSE
        elif PropertyTypeRegexp.CONDO.match(value):
            return PropertyType.CONDO
    return None


def allTypeToFloatRow(_, value):
    return allTypeToFloat(value)


def allTypeToIntRow(_, value):
    return allTypeToInt(value)


def shallDrop(row):
    try:
        if (row['saletp-b'] == 0):
            if (row['lp-n'] == 0) | (row['lp'] is None) | (row['lp'] < SALE_PRICE_LOWER_LIMIT):
                return True
        if (row['saletp-b'] == 1):
            if (row['lpr-n'] == 0) | (row['lpr'] is None) | (row['lpr'] > RENT_PRICE_UPPER_LIMIT):
                return True
        if (row['lst'] in ['Sld', 'Lsd']):
            if (row['sp-n'] == 0) | (row['sp'] is None):
                return True
    except Exception as e:
        logger.error(row)
        logger.error('shallDrop', e)
    return False


def taxYearRow(row, value):
    yr = allTypeToInt(value)
    if yr is not None:
        if 200 <= yr < 300:
            yr = yr % 100 + 2000
        if yr < 200:
            yr = yr + 2000
        if 1990 <= yr <= datetime.datetime.now().year:
            return yr
    return yearOfByField(row, 'onD', -183)


def laundryLevelRow(_, value):
    return getLevel(value)


def petsRow(_, value):
    value = str(value)[0]
    if value == 'Y':
        return 2
    elif value == 'R':
        return 1
    elif value == 'N':
        return 0
    else:
        return 2  # unknown


def balconyRow(_, value):
    return balconyType.get(value, 0)


SUFFIXES = {
    '-n': 'Number',
    '-c': 'Category Number',
    '-b': 'Binary 0/1',
    '-l': 'String Label',
}


class Preprocessor(TransformerMixin, BaseEstimator):
    """ Transforms raw training and prediction data
    To build root transformer, use TRAIN mode and fit with full dataset(columns and rows). 
    To transform predict data, use PREDICT mode and fit with training/predict data.
    If columns changed, use PREDICT mode and fit with training data. It can transform both training and predict data.
    Different column sets need different preprocessors.

    Transforming steps:
    -. drop na columns if in training mode, return error if in prediction mode.
    -. convert binary saletp to 0 or 1, column name as 'saletp-b'
    -. convert ptype2 to single value. column name as 'ptype2-l'
    -. convert binary cols. column name as '-b'
    -. convert categorical columns to integers. column name as '-c'
    -. fill numeric columns to default values. column name as '-n'
    -. filter lat/lng to the range of [-180, 180] and drop null rows.
    """
    # binary use index as value, default 0
    cols_binary: dict = {
        'status':       ['U', 'A'],
        'den_fr':       ['N', 'Y'],
        'ens_lndry':    ['N', 'Y'],
        'cac_inc':      ['N', 'Y'],
        'comel_inc':    ['N', 'Y'],
        'heat_inc':     ['N', 'Y'],
        'prkg_inc':     ['N', 'Y'],
        'hydro_inc':    ['N', 'Y'],
        'water_inc':    ['N', 'Y'],
        'insur_bldg':   ['N', 'Y'],
        'tv':           ['N', 'Y'],
        'all_inc':      ['N', 'Y'],
        'furnished':    ['N', 'Y'],
        'retirement':   ['N', 'Y'],
        'pvt_ent':      ['N', 'Y'],
    }
    cols_label: dict = {  # na DROP means to remove the rows without label
        'lst':      {'na': UNKNOWN},
        'prov':     {'na': DROP},
        'area':     {'na': UNKNOWN},
        'city':     {'na': DROP},
        'cmty':     {'na': UNKNOWN},
        'st':       {'na': UNKNOWN},
        'zip':      {'na': UNKNOWN},
        'rltr':     {'na': UNKNOWN},
        #        'saletp':   {'na': DROP},
        'ptype2-l': {'na': DROP},
        'pstyl':    {'na': UNKNOWN},  # 131 types
        'ptp':      {'na': UNKNOWN},  # ptpType
        'zone':     {'na': UNKNOWN},
    }
    cols_array_label: dict = {
        'constr':   {'map': constrType, 'strType': False},
        'feat':     {'map': featType, 'strType': False},
        'bsmt':     {'map': bsmtType, 'strType': False},
        'fuel':     {'map': fuelType, 'strType': False},
        'laundry':  {'map': laundryType, 'strType': False},
        'park_desig': {'map': parkingDesignationType, 'strType': False},
        'ac':       {'map': acType, 'strType': True},
        'gatp':     {'map': garageType, 'strType': True},
        'lkr':      {'map': lockerType, 'strType': True},
        'heat':     {'map': heatType, 'strType': True},
        'fce':      {'map': exposureType, 'strType': True},
        'park_fac':  {'map': parkingFacilityType, 'strType': True},
    }
    cols_numeric: dict = {
        'lat':      {'na': DROP},
        'lng':      {'na': DROP},
        #        'st_num':   {'na': 0},
        'mfee':     {'na': 0},
        'tbdrms':   {'na': 0},
        'bdrms':    {'na': 0},
        'br_plus':  {'na': 0},
        'bthrms':   {'na': 0},
        'kch':      {'na': 0},
        'kch_plus': {'na': 0},
        'tgr':      {'na': 0},
        'gr':       {'na': 0},
        'lp':       {'na': 0},
        'lpr':      {'na': 0},
        'sp':       {'na': 0},
        'depth':    {'na': 0},
        'flt':      {'na': 0},
        # 'onD':      {'na': DROP},
        # 'offD':     {'na': 0},
    }

    # ----- Special cases: Done ---------
    cols_todo: dict = {
        'ptype':    {'na': 'r'},
    }

    # Done.
    cols_special: dict = {
        'lp':       {'na': DROP},
        'lpr':      {'na': DROP},
        'sp':       {'na': DROP},
        # extract number parts from street number
        'st_num':   {'to': 'st_num-n'},  # Done.
        'sqft':     {'to': 'sqft-n'},  # Done.
        'rmSqft':   {'to': 'sqft-n'},  # rmSqft or sqft estimator
        # from bltYr or rmBltYr or bltYr estimator
        'bltYr':    {'to': 'built_yr-n'},  # Done.
        'rmBltYr':  {'to': 'built_yr-n'},  # rmBltYr or bltYr estimator
        'ptype2':   {'to': 'ptype2-l'},  # ptype2. Done
        'ac':       {'to': 'ac-n'},  # ac
        'balcony':  {'to': 'balcony-n'},  # Done
        'laundry_lev': {'na': NONE},  # Done
        'pets':     {'na': UNKNOWN},  # Done
    }

    # Done
    cols_structured_todo: list[str] = [
        'rms',  # get primary bedroom dimensions and area, sum of all bedrooms deminsions, sum of all bedrooms area
        'bths',  # get bath numbers on each level => l0-l3 * number of bathrooms; l0-l3 * pices total
    ]

    cols_forsale_in_models: dict = {
        'tax':      {'na': MEAN},  # Done
        # the first half year counted as previous tax year
        # Done
        'taxyr':    {'na': lambda row: yearOfByField(row, 'onD', -183)},
    }
    cols_forsale_house_in_models: dict = {
        # MEAN of all depths when Detached/Semi-Detached/Freehold Townhouse
        'depth':    {'na': MEAN},  # Done
        # MEAN of all flt when Detached/Semi-Detached/Freehold Townhouse
        'flt':      {'na': MEAN},  # Done
    }
    cols_not_used: list[str] = [
        'la',  # la id : la.agnt[].id
        'la2',  # la2 id: la2.agnt[].id
        'schools',  # get 3 school names and ratings, rankings
    ]
    cols_condo: list[str] = [
        'unt'  # unit storey, total storey, percentage of total storey
    ]

    def __init__(
        self,
        collection_prefix: str = 'ml_'
    ):
        self.collection_prefix: str = collection_prefix

    def get_feature_columns(
        self,
        all_cols: list[str] = None,
    ) -> list[str]:
        """Get the feature columns for the model.

        Returns
        -------
        list[str]
            feature columns
        """
        if all_cols is None:
            if not hasattr(self.customTransformer) or self.customTransformer is None:
                raise ValueError('No transformer built yet')
        else:
            self.build_transformer(all_cols)
        return self.customTransformer.get_feature_names()

    def build_transformer(self, all_cols):
        """Build the transformers.
           The transformers need to work with less columns when the predictions has less data.

            Parameters
            ----------
            all_cols : list of strings
                all columns to transform

        """
        logger.info('Building transformers')
        all_cols = [*all_cols]

        colTransformerParams = [
            ('saletp-b', binarySaletpByRow, 'saletp', 'saletp-b'),
            ('ptype2-l', ptype2SingleValue, 'ptype2', 'ptype2-l', True),
        ]
        all_cols.append('saletp-b')
        all_cols.append('ptype2-l')
        # custom transformers
        if 'onD' in all_cols:
            datesTransformer = DatesTransformer(all_cols)
            colTransformerParams.append(
                ('onD', datesTransformer))
            all_cols.extend(datesTransformer.get_feature_names_out())
        if 'pets' in all_cols:
            colTransformerParams.append(('pets', petsRow, 'pets', 'pets-n'))
        if 'laundry_lev' in all_cols:
            colTransformerParams.append(
                ('laundry_lev', laundryLevelRow, 'laundry_lev', 'laundry_lev-n'))
        if 'balcony' in all_cols:
            colTransformerParams.append(
                ('balcony', balconyRow, 'balcony', 'balcony-n'))
        if 'flt' in all_cols:
            colTransformerParams.append(
                ('flt', allTypeToFloatRow, 'flt', 'flt-n'))
        if 'depth' in all_cols:
            colTransformerParams.append(
                ('depth', allTypeToFloatRow, 'depth', 'depth-n'))
        if 'tax' in all_cols:
            colTransformerParams.append(
                ('tax', allTypeToFloatRow, 'tax', 'tax-n'))
            colTransformerParams.append(
                ('taxyr', taxYearRow, 'taxyr', 'taxyr-n'))
        if ('bltYr' in all_cols) or ('rmBltYr' in all_cols):
            colTransformerParams.append(('bltYr', SelectColumnTransformer(
                new_col='bltYr-n', columns=['bltYr', 'rmBltYr'], func=stringToInt, as_na_value=None)))
        if ('sqft' in all_cols) or ('rmSqft' in all_cols):
            colTransformerParams.append(('sqft', SelectColumnTransformer(
                new_col='sqft-n', columns=['sqft', 'rmSqft'], func=stringToInt, as_na_value=None)))
        if 'st_num' in all_cols:
            colTransformerParams.append(
                ('st_num', allTypeToIntRow, 'st_num', 'st_num-n'))
        # array labels
        for k, v in self.cols_array_label.items():
            if k in all_cols:
                if v['strType']:
                    transformer = DbOneHotArrayEncodingTransformer(
                        col=k,
                        map=v['map'],
                        sufix='-b',
                        na_value=None,
                        collection=self.label_collection,
                    )
                else:
                    transformer = DbOneHotArrayEncodingTransformer(
                        col=k,
                        map=v['map'],
                        sufix='-b',
                        # na_value=None,
                        # collection=self.label_collection,
                    )
                colTransformerParams.append((f'{k}_x', transformer))
                all_cols.extend(transformer.get_feature_names_out())
        # binary columns
        for k, v in self.cols_binary.items():
            if k in all_cols:
                colTransformerParams.append(
                    (f'{k}-b', BinaryTransformer(v, k), k, f'{k}-b'))
                all_cols.append(f'{k}-b')
        # categorical columns
        for k, v in self.cols_label.items():
            if k in all_cols:
                colTransformerParams.append(
                    (f'{k}-c', DbLabelTransformer(
                        col=k,
                        na_value=v['na'],
                        collection=self.label_collection,
                    ), k, f'{k}-c'))
                all_cols.append(f'{k}-c')
        # numerical columns
        for k, v in self.cols_numeric.items():
            if k in all_cols:
                colTransformerParams.append(
                    (f'{k}-n', DbNumericTransformer(
                        self.number_collection,
                        col=k,
                        na_value=v['na'],), k, f'{k}-n'))
                all_cols.append(f'{k}-n')
        if 'st_num' in all_cols:
            stNumStTransformer = StNumStTransformer()
            colTransformerParams.append(
                ('st_num-st', stNumStTransformer))
            all_cols.extend(stNumStTransformer.get_feature_names_out())
        # rms and bths
        if 'rms' in all_cols:
            colTransformerParams.append(('rms', RmsTransformer()))
        if 'bths' in all_cols:
            colTransformerParams.append(('bths', BthsTransformer()))
        # drop rows: this operation has to be done outside of the pipeline
        # the drop operation is dependent on the model's usage of columns
        drop_na_cols = []
        for k, v in chain(self.cols_label.items(), self.cols_numeric.items()):
            if (k in all_cols) and (v['na'] is DROP):
                drop_na_cols.append(k)
        self.drop_na_cols_ = drop_na_cols
        colTransformerParams.append(
            ('drop_na', DropRowTransformer(drop_cols=drop_na_cols))
        )
        colTransformerParams.append(
            ('drop_check', DropRowTransformer(drop_func=shallDrop))
        )

        # create the pipeline
        self.customTransformer = SimpleColumnTransformer(
            colTransformerParams)
        return self.customTransformer

    def fit(self, Xdf: pd.DataFrame, y=None):
        """A reference implementation of a fitting function for a transformer.
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
        self.label_collection = self.collection_prefix + 'label'
        self.number_collection = self.collection_prefix + 'number'

        self.build_transformer(Xdf.columns)
        self.customTransformer.fit(Xdf, y)
        self.n_features_ = Xdf.shape[1]
        return self

    def transform(self, Xdf: pd.DataFrame):
        """Transform the dataframe to the baseline format.

            Parameters
            ----------
            df : pd.DataFrame
                the dataframe to transform

        """
        if self.n_features_ is None:
            raise ValueError('The transformer has not been fitted yet.')

        # if self.customTransformer is None:
        #     self.build_transformer(Xdf.columns)
        #     self.customTransformer.fit(Xdf)
        logger.info('Transforming to baseline format')
        #pd.set_option('mode.chained_assignment', None)
        Xdf = self.customTransformer.transform(Xdf)
        #pd.set_option('mode.chained_assignment', 'warn')
        return Xdf
