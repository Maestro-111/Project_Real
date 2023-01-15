import numbers
import re
from .base_cfg import BaseCfg
import pandas as pd
import numpy as np
from math import isnan
from datetime import datetime, timezone
UTC_OFFSET_TIMEDELTA = datetime.utcnow() - datetime.now()

logger = BaseCfg.getLogger(__name__)


def selectFromTwo(a, b):
    if (a is not None) and (not isnan(a)):
        return a
    return b


def utc_to_local(utc_dt):
    return utc_dt.replace(tzinfo=timezone.utc).astimezone(tz=None)


def local_to_utc(local_dt):
    return local_dt + UTC_OFFSET_TIMEDELTA


def get_utc_datetime_from_str(str_datetime):
    if len(str_datetime) == 10:
        str_datetime = str_datetime + ' 00:00:00'
    return datetime.strptime(str_datetime, '%Y-%m-%d %H:%M:%S')


def print_dateframe(df):
    # more options can be specified also
    with pd.option_context('display.max_rows', BaseCfg.pd_max_rows, 'display.max_columns', BaseCfg.pd_max_columns):
        print(df)


def index_min(values):
    # min(range(len(values)), key=values.__getitem__)
    return np.argmin(values)


def print_all_columns(df):
    pd.set_option('display.max_columns', None)
    print(df.columns.values, df.columns.values.shape)


def allTypeToFloat(value):
    v_type = type(value)
    if v_type is int:
        return float(value)
    if v_type is float:
        if isnan(value):
            return None
        return value
    if v_type is str:
        num_array = re.findall(r'\d+(?:\.\d+)?', value)
        value = ''.join(num_array)
        if len(value) > 0:
            if value.isnumeric():
                return float(value)
            return float(num_array[0])
    if v_type is list:
        return allTypeToFloat(value[0])
    return None


def allTypeToInt(value):
    v_type = type(value)
    if v_type is int:
        return value
    if v_type is float:
        if isnan(value):
            return None
        return int(value)
    if v_type is str:
        int_array = re.findall(r'\d+(?:\.\d+)?', value)
        value = ''.join(int_array)
        if len(value) > 0:
            return int(float(value))
    if v_type is list:
        return allTypeToInt(value[0])
    return None


reAllNumber = re.compile(r'^\s*\d+(?:\.\d+)?\s*$')


def stringToFloat(value):
    if isinstance(value, str) and reAllNumber.match(value):
        return float(value)
    if (isinstance(value, float) or isinstance(value, int)) and not isnan(value):
        return float(value)
    return None


def stringToInt(value):
    if isinstance(value, str) and reAllNumber.match(value):
        return int(value)
    if (isinstance(value, float) or isinstance(value, int)) and not isnan(value):
        return int(value)
    return None


def dateFromNum(dayNum):
    if isnan(dayNum):
        return None
    dayNum = int(dayNum)
    day = dayNum % 100
    year = dayNum // 10000
    month = (dayNum // 100) - (year * 100)
    if month < 1:
        month = 1
        print(f'month < 1 : {dayNum}')
    if month > 12:
        if dayNum < 2000001:  # there are numbers like 2017125
            year = dayNum // 1000
            month = (dayNum % 1000) // 100
        if month > 12:
            month = 12
            print(f'month > 12 : {dayNum}')
    return datetime(year, month, day, 0, 0)


def dateToNum(date: datetime = datetime.today()):
    return date.year * 10000 + date.month * 100 + date.day


def isNanOrNone(value):
    if value is None:
        return True
    if (value == 'nan') or (value == ''):
        return True
    if isinstance(value, numbers.Number) and isnan(value):
        return True
    return False


def getUniqueLabels(X: pd.Series) -> list:
    return X.unique().tolist()


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


def debug(fn):
    def wrapper(*args, **kwargs):
        logger.debug(f"Invoking {fn.__name__}")
        logger.debug(f"  args: {args}")
        logger.debug(f"  kwargs: {kwargs}")
        result = fn(*args, **kwargs)
        logger.debug(f"  returned {result}")
        return result
    return wrapper


def columnValues(df: pd.DataFrame, col: str) -> list:
    return df[col].unique().tolist()


def getRoundFunction(n=1):
    def roundByN(value):
        if isnan(value):
            return value
        return int(round(value / n) * n)
    return roundByN


def roundByN(value, n=1):
    if isnan(value):
        return value
    return int(round(value / n) * n)


def expendList(list1, list2):
    for item in list2:
        if item not in list1:
            list1.append(item)
    return list1
