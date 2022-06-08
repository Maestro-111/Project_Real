
import time
from base.base_cfg import BaseCfg
from prop.estimate_scale import EstimateScale
from base.mongo import MongoDB
from base.util import get_utc_datetime_from_str, print_dateframe
from datetime import datetime, timedelta
logger = BaseCfg.getLogger('ml.data_reader')


def dateToInt(date):
    """Convert date to int"""
    return int(datetime.strftime(date, '%Y%m%d'))


def read_data(scale: EstimateScale, col_list: list[str], date_span: int = 180, query: dict = {}):
    """Read the data. Default date_span is 180 days"""
    if not isinstance(scale, EstimateScale):
        raise ValueError('scale must be an instance of EstimateScale')
    dateTo = dateToInt(scale.datePoint)
    dateFrom = dateToInt(scale.datePoint - timedelta(days=date_span))
    geoQuery = scale.get_geo_query()
    typeQuery = scale.get_type_query()
    filter = {
        'ptype': 'r',
        'onD': {
            '$gte': dateFrom,
            '$lte': dateTo}}
    filter = {**filter, **geoQuery, **typeQuery,  **query}
    return read_data_by_query(filter, col_list)


def read_data_by_query(query: dict, col_list: list[str], mongodb: MongoDB = None):
    # use mongo connection to read data
    if mongodb is None:
        mongodb = MongoDB()
    print('Query:', str(query))
    start_time = time.time()
    result = mongodb.load_data('properties', col_list, query)
    if BaseCfg.isDebug():
        print('columns:', result.columns, 'shape:', result.shape)
        print_dateframe(result)
    end_time = time.time()
    print('data shape:', str(result.shape),
          'used:', str(end_time - start_time))
    return result
