import re
from .base_cfg import BaseCfg
import pandas as pd
import numpy as np
from math import isnan
from datetime import datetime, timezone
UTC_OFFSET_TIMEDELTA = datetime.utcnow() - datetime.now()


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
        int_array = re.findall(r'\d+(?:\.\d+)?', value)
        value = ''.join(int_array)
        if len(value) > 0:
            return float(value)
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
