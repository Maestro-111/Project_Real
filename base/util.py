from .base_cfg import BaseCfg
import pandas as pd
import numpy as np
from datetime import datetime, timezone
UTC_OFFSET_TIMEDELTA = datetime.utcnow() - datetime.now()


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
