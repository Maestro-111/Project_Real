

import time
from base.base_cfg import BaseCfg
from numpy import NaN
from predictor.writeback_mixin import WriteBackMixin
from data.estimate_scale import EstimateScale
from base.mongo import MongoDB
from base.util import debug, get_utc_datetime_from_str, getUniqueLabels, print_dateframe
from datetime import datetime, timedelta
import pandas as pd
from transformer.preprocessor import Preprocessor
from typing import Union
import os

logger = BaseCfg.getLogger(__name__)

PROV_CITY_TO_AREA = {}
PROV_CITY_TO_AREA_FILE = 'data/prov_city_to_area.csv'


def readProvCityToArea():
    """Read province-city to area mapping from csv file"""
    global PROV_CITY_TO_AREA
    # check file exists
    if not (os.path.exists(PROV_CITY_TO_AREA_FILE) and os.path.isfile(PROV_CITY_TO_AREA_FILE)):
        return
    df = pd.read_csv(PROV_CITY_TO_AREA_FILE)
    for index, row in df.iterrows():
        PROV_CITY_TO_AREA[(row['province'], row['city'])] = row['area']

readProvCityToArea()

def writeProvCityToArea():
    """Write province-city to area mapping to csv file"""
    df = pd.DataFrame(columns=['province', 'city', 'area'])
    for key, value in PROV_CITY_TO_AREA.items():
        df = df.append(
            {'province': key[0], 'city': key[1], 'area': value}, ignore_index=True)
    df.to_csv(PROV_CITY_TO_AREA_FILE, index=False)


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
    """ use mongo connection to read data from mongodb
    """
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
    """DataSource class to store data sources.
    Dataframes:
        df_raw: the raw data frame
        df_transformed: the transformed data frame
        df_grouped: the grouped data frame from transformed data frame

    Parameters:
    =================
    scale: EstimateScale
        the scale to read data from mongodb
    query: dict, optional
        the extra query combined to the scale to read data from mongodb
    col_list: list[str]
        the columns to read from mongodb
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

    def __init__(
        self,
        scale: EstimateScale,
        query: dict = None,
        col_list: list[str] = None
    ):
        """Initialize DataSource object.
        """
        self.scale = scale
        self.query = query
        self.col_list = col_list if col_list else DataSource.all_data_col_list
        self.df_raw = None
        self.df_transformed = None
        self.df_grouped = None
        self._query = {**query, **self.scale.get_query()}

    def __str__(self):
        return f'DataSource: {self._query}'

    def load_raw_data(self):
        """Load raw data from mongodb"""
        self.df_raw = read_data_by_query(
            self._query, self.col_list)
        self._build_prov_city_to_area()
        self._build_scale_tree()

    def _build_prov_city_to_area(self):
        """Build PROV_CITY_TO_AREA dict from df_raw"""
        global PROV_CITY_TO_AREA
        for prov, area, city in self.df_raw[['prov', 'area', 'city']].values:
            PROV_CITY_TO_AREA[(prov, city)] = area
        writeProvCityToArea()
        return PROV_CITY_TO_AREA

    def load_df_grouped(
        self,
        id_list: list[str] = None,
        preprocessor: Preprocessor = None,
    ) -> pd.DataFrame:
        """Load new data from mongodb.
        This function is used to load new data from mongodb for prediction.
        """
        global PROV_CITY_TO_AREA
        query = {'_id': {'$in': id_list}}
        df_raw_to_predict = read_data_by_query(query, self.col_list)
        # fill missing area
        df_raw_to_predict['area'] = df_raw_to_predict.apply(
            lambda row: PROV_CITY_TO_AREA.get((row['prov'], row['city']), 'Other'), axis=1)
        logger.info(PROV_CITY_TO_AREA)
        logger.info(df_raw_to_predict[['prov', 'area', 'city']])
        df_transformed_to_predict = preprocessor.transform(df_raw_to_predict)
        # groupby and reindex by EstimateScale
        df_grouped_to_predict = df_transformed_to_predict.set_index([
            'saletp-b', 'ptype2-l',
            'prov', 'area', 'city',
            '_id',
        ]).sort_index(
            level=[0, 1, 2, 3, 4],
            ascending=[True, True, True, True, True],
            inplace=False,
        )
        return df_grouped_to_predict

    def concat_df_grouped(self, df_grouped: pd.DataFrame):
        """Concat grouped data with new data.
        """
        self.df_grouped = pd.concat(
            [self.df_grouped, df_grouped], axis=0)

    def _build_scale_tree(self):
        """Build scale tree from raw data."""
        provs = self.df_raw['prov'].unique().tolist()
        areas = self.df_raw['area'].unique().tolist()
        cities = self.df_raw['city'].unique().tolist()
        logger.debug(
            f'Build Scale Tree. \n provs: {provs} \n areas: {areas} \n cities: {cities}')
        self.scale.buildAllSubScales(provs, areas, cities)

    def transform_data(self, preprocessor: Preprocessor = None):
        """ Transform data with preprocessor then group them.
        If raw data is not loaded, load it first.
        Args:
            preprocessor (Preprocessor, optional): Defaults to None.
                provide fit_transform method to transform df_raw to df_transformed. 

        Returns:
            DataFrame: transformed DataFrame.
        """
        if self.df_raw is None:
            self.load_raw_data()
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

    # @debug
    def get_df(
        self,
        scale: EstimateScale,
        cols: list[str] = None,
        date_span: int = 180,
        need_raw: bool = False,
        suffix_list: list[str] = None,
        copy: bool = False,
        sample_size: int = None,
        filter_func: (pd.Series) = None,
        numeric_columns_only: bool = False,
        prefer_estimated: bool = False,
        df_grouped: pd.DataFrame = None,
    ):
        """Get dataframe from stored data.
        """
        if date_span is None:
            date_span = 180
        if suffix_list is None:
            suffix_list = ['-b', '-n', '-c']
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
        if df_grouped is None:
            df_grouped = self.df_grouped
        rd = df_grouped.loc[tuple(slices), :]
        logger.debug(f'{slices} {len(df_grouped.index)}=>{len(rd.index)}')
        if len(rd.index) == 0:
            return None
        # onD:
        if date_span > 0:
            rd = rd.loc[rd.onD.between(
                dateToInt(scale.datePoint - timedelta(days=date_span)),
                dateToInt(scale.datePoint)
            )]
        logger.debug(
            f'{scale.datePoint-timedelta(days=date_span)}-{scale.datePoint} {len(rd.index)}')
        # filter data by filter_func
        rd = rd.loc[rd.apply(filter_func, axis=1)] if filter_func else rd
        logger.debug(f'after filter_func {len(rd.index)}')
        if len(rd.index) == 0:
            return None
        # sample data
        if sample_size is not None and sample_size < rd.shape[0]:
            rd = rd.sample(n=sample_size, random_state=1)
        # select columns from cols
        existing_cols = rd.columns.tolist()
        if cols is not None:
            columns = []
            for col in cols:
                found = False
                # ['-b', '-n', '-c', '-l', ]:  '-l' is only in 'ptype2-l'
                for suffix in suffix_list:
                    for c in existing_cols:
                        if c.startswith(col) and c.endswith(suffix):
                            columns.append(c)
                            # logger.debug(f'column[{col}] found as [{c}]')
                            found = True
                if (col in existing_cols) and (need_raw or not found):
                    columns.append(col)
                    # logger.debug(f'column[{col}] found as [{c}] (raw)')
                    found = True
                if not found:
                    logger.debug(f'column[{col}] not found')
            columns = list(dict.fromkeys(columns))  # remove duplicates
        else:
            columns = existing_cols
        rd = rd.loc[:, columns]
        # if numeric_columns_only is True, only keep numeric columns
        if numeric_columns_only:
            numeric_columns = self.get_numeric_columns(
                df=rd, prefer_estimated=prefer_estimated)
            rd = rd.loc[:, numeric_columns]
        # return dataframe or copy
        return rd.copy() if copy else rd

    def get_numeric_columns(
        self,
        df: pd.DataFrame,
        exclude_columns: list[str] = None,
        prefer_estimated: bool = False,
    ) -> list[str]:
        """Get the numeric columns

        Parameters
        ==========
        df: pd.DataFrame
        exclude_columns: list[str]
            columns to be excluded from the result
        prefer_estimated_value: bool
            if True, return the estimated value column if it exists,
            and remove the original column.
        """
        if exclude_columns is None:
            exclude_columns = []
        numeric_columns = []
        for col in df.columns:
            try:
                if (col not in exclude_columns) and \
                    (df[col].dtype == 'float64' or
                     df[col].dtype == 'int64'):
                    numeric_columns.append(col)
            except Exception as e:
                self.logger.error(
                    f'Error in getting numeric column:{col} error:{e}')
        if prefer_estimated:
            new_cols = numeric_columns.copy()
            for col in numeric_columns:
                if col.endswith('-e'):
                    for suffix in ['-c', '-n', '-b']:
                        orig_col = col[:-2] + suffix
                        if orig_col in numeric_columns:
                            new_cols.remove(orig_col)
            numeric_columns = new_cols
        return numeric_columns

    def writeback(
        self,
        col: Union[str, list[str]],
        y: Union[pd.Series, pd.DataFrame],
        df_grouped: pd.DataFrame = None,
    ) -> None:
        """write y to df_grouped
        """
        if df_grouped is None:
            df_grouped = self.df_grouped
        yIsSeries = False
        if isinstance(y, pd.Series):
            yIsSeries = True
        if isinstance(col, str):
            col = [col]
        if not isinstance(col, list):
            raise Exception(
                f'col must be str or list[str], but got {type(col)}')
        for c in col:
            if c not in df_grouped.columns:
                df_grouped.loc[:, c] = NaN
            if yIsSeries:
                df_grouped.loc[y.index, c] = y
            else:
                df_grouped.loc[y.index, c] = y.loc[:, c]
        return df_grouped


class TrendDataSource:
    """TODO: Trend Data Source.
    Provide dataframe for trend analysis.
    Columns:
    indexes:
        saletp-b, ptype2-l, prov, area, city, periodId
    keys:
        year, month, week,
    first level:
        new, sold, off, pureNew, pc,
        askPriceAvg, soldPriceAvg,
        soldDomAvg, offDomAvg,
        askPerTaxAvg, soldPerTaxAvg,
        startAskPriceAvg, endAskPriceAvg,
        startAvailAvg, endAvailAvg,
    secondary level:
        pureNewPerNew, pureNewPerSold, pureNewPerOff,
        soldPerNew, soldPerOff,
        soldDomAvg, soldDomMedian,
        offDomAvg, offDomMedian,
        startAvgPrice, endAvgPrice, priceDiffAvg,
        startMedianPrice, endMedianPrice, priceDiffMedian, 
        pcPerNewAvg, priceChangePerNewMedian,
        priceSoldPerAsk, priceChangedPercent, priceChanged,
    date features:
        periodWeek, periodMonth, periodQuarter, periodYear,

    """

    def __init__(self) -> None:
        pass
