

import time
from base.base_cfg import BaseCfg
from numpy import NaN
from predictor.writeback_mixin import WriteBackMixin
from prop.estimate_scale import EstimateScale
from base.mongo import MongoDB
from base.util import get_utc_datetime_from_str, print_dateframe
from datetime import datetime, timedelta
import pandas as pd
from transformer.preprocessor import Preprocessor
logger = BaseCfg.getLogger(__name__)


def dateToInt(date):
    """Convert date to int"""
    return int(datetime.strftime(date, '%Y%m%d'))


def read_data(
    scale: EstimateScale,
    col_list: list[str],
    date_span: int = 180,
    query: dict = {}
):
    """Read the data. Default date_span is 180 days"""
    if not isinstance(scale, EstimateScale):
        raise ValueError('scale must be an instance of EstimateScale')
    dateTo = dateToInt(scale.datePoint)
    dateFrom = dateToInt(scale.datePoint - timedelta(days=date_span))
    geoQuery = scale.get_geo_query()
    typeQuery = scale.get_type_query()
    saletpQuery = scale.get_saletp_query()
    filter = {
        'ptype': 'r',
        'onD': {
            '$gte': dateFrom,
            '$lte': dateTo}}
    filter = {**filter, **geoQuery, **typeQuery, **saletpQuery,  **query}
    return read_data_by_query(filter, col_list)


def read_data_by_query(
    query: dict,
    col_list: list[str],
    mongodb: MongoDB = None
):
    # use mongo connection to read data
    if mongodb is None:
        mongodb = MongoDB()
    logger.info(f'Mongo Query: {query}')
    start_time = time.time()
    result = mongodb.load_data('properties', col_list, query)
    if BaseCfg.isDebug():
        logger.debug('columns: {result.columns} shape: {result.shape}')
        print_dateframe(result)
    end_time = time.time()
    logger.info(
        f'Result data shape:{result.shape}; used: {end_time - start_time}s')
    return result


class DataSource:
    """DataSource class.

    This class is used to store data sources.
    """

    all_data_query = {
        # test: '_id': {'$in': ['TRBE5467506', 'TRBC5467511']},
        # only use data after 2021 for test purpose
        'onD': {'$gt': 20200101},
        'ptype': 'r',
        'prov': 'ON',
        'area': {
            '$in': [
                'Toronto', 'York', 'Halton', 'Durham',
                'Peel', 'Hamilton', 'Simcoe',
            ]
        },
    }
    all_data_col_list: list[str] = [
        '_id', 'onD', 'offD', 'sldd', 'lst', 'status',
        'prov', 'area', 'city', 'cmty', 'addr', 'uaddr',
        'st', 'st_num', 'lat', 'lng', 'unt', 'zip',
        'ptype', 'ptype2', 'saletp', 'pstyl', 'ptp',
        'lp', 'lpr', 'sp', 'tax', 'taxyr', 'mfee',
        'bdrms', 'tbdrms', 'br_plus', 'bthrms', 'kch', 'kch_plus',
        'bths', 'rms', 'bsmt', 'schools', 'zone',
        'gr', 'tgr', 'gatp',
        'depth', 'flt',
        'heat', 'feat', 'constr', 'balcony', 'ac',
        'den_fr', 'ens_lndry', 'fce', 'lkr',
        'sqft', 'rmSqft', 'bltYr', 'rmBltYr',
        'cac_inc', 'comel_inc', 'heat_inc', 'prkg_inc',
        'hydro_inc', 'water_inc', 'all_inc', 'pvt_ent',
        'insur_bldg', 'tv',
        'pets', 'laundry', 'laundry_lev',
        'daddr', 'commuId', 'park_fac',
        'comm', 'rltr', 'la', 'la2',
    ]

    def __init__(self, query: dict = None, col_list: list[str] = None):
        """Initialize DataSource object.

        Args:
          name: Name of the data source.
        """
        self.query = query if query else DataSource.all_data_query
        self.col_list = col_list if col_list else DataSource.all_data_col_list
        self.df_raw = None
        self.df_transformed = None

    def __str__(self):
        return 'DataSource: {}'.format(self.query)

    def load_data(self, preprocessor: Preprocessor = None):
        self.df_raw = read_data_by_query(
            self.query, self.col_list)
        if preprocessor is not None:
            self.df_transformed = preprocessor.fit_transform(self.df_raw)
            # groupby and reindex by EstimateScale
            self.df_grouped = self.df_transformed.set_index([
                'saletp-b', 'ptype2-l',
                'prov', 'area', 'city',
                '_id',
            ]).sort_index(
                level=[0, 1, 2, 3, 4],
                ascending=[True, True, True, True, True],
                inplace=False,
            )
            return self.df_grouped
        return self.df_raw

    def get_df(
        self,
        scale: EstimateScale,
        cols: list[str],
        date_span: int = 180,
        need_raw: bool = False,
        suffix_list: list[str] = ['-b', '-n', '-c'],
        copy: bool = False,
        sample_size: int = None,
        filter_func: (pd.Series) = None,
    ):
        """Get dataframe from stored data.
        """
        if date_span is None:
            date_span = 180
        slices = []
        # filter data by scale and date_span
        # saletp_b
        if scale.sale is True:
            slices.append(slice(0, 0))
        elif scale.sale is False:
            slices.append(slice(1, 1))
        else:  # scale.sale is None
            slices.append(slice(None))
        # ptype2_l
        if scale.propType is not None:
            slices.append(slice(scale.propType, scale.propType))
        else:
            slices.append(slice(None))
        # prov, area, city
        if scale.prov is not None:
            slices.append(slice(scale.prov, scale.prov))
        else:
            slices.append(slice(None))
        if scale.area is not None:
            slices.append(slice(scale.area, scale.area))
        else:
            slices.append(slice(None))
        if scale.city is not None:
            slices.append(slice(scale.city, scale.city))
        else:
            slices.append(slice(None))
        slices.append(slice(None))
        rd = self.df_grouped.loc[tuple(slices), :]
        logger.debug(f'{slices} {len(self.df_grouped.index)}=>{len(rd.index)}')
        # onD:
        rd = rd.loc[rd.onD.between(
            dateToInt(scale.datePoint - timedelta(days=date_span)),
            dateToInt(scale.datePoint)
        )]
        logger.debug(
            f'{scale.datePoint-timedelta(days=date_span)}-{scale.datePoint} {len(rd.index)}')
        # filter data by filter_func
        rd = rd.loc[rd.apply(filter_func, axis=1)] if filter_func else rd
        logger.debug(f'after filter_func {len(rd.index)}')
        # sample data
        if sample_size is not None and sample_size < rd.shape[0]:
            rd = rd.sample(n=sample_size, random_state=1)
        # select columns from cols
        existing_cols = rd.columns.tolist()
        columns = []
        for col in cols:
            found = False
            # ['-b', '-n', '-c', '-l', ]:  '-l' is only in 'ptype2-l'
            for suffix in suffix_list:
                for c in existing_cols:
                    if c.startswith(col) and c.endswith(suffix):
                        columns.append(c)
                        found = True
            if (col in existing_cols) and (need_raw or not found):
                columns.append(col)
                found = True
            if not found:
                print(f'column[{col}] not found')
        columns = list(dict.fromkeys(columns))  # remove duplicates
        rd = rd.loc[:, columns]
        # return dataframe or copy
        return rd.copy() if copy else rd

    def writeback(
        self,
        predictor: WriteBackMixin,
        new_col: str,
        scale: EstimateScale,
        cols: list[str],
        date_span: int = 180,
        need_raw: bool = False,
        suffix_list: list[str] = ['-b', '-n', '-c'],
        orig_col: str = None,
    ):
        """Writeback dataframe to df_grouped.
        """
        if new_col not in self.df_grouped.columns:
            # need to set dtype as number
            self.df_grouped.loc[:, new_col] = NaN
        # TODO: why only get part of the matched df
        df = self.get_df(
            scale, cols, date_span, need_raw, suffix_list, copy=False)
        y = predictor.get_writeback(df, orig_col)
        logger.debug(
            f'writeback df:{df.shape} y({new_col}):{len(y)} type:{y.dtype}')
        df.loc[:, new_col] = y
        self.df_grouped.update(df.loc[:, new_col])
        # print(self.df_grouped)
