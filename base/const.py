from enum import Enum


UNKNOWN = 'unknown'
NONE = 'none'
CUSTOM = 'custom'
MEAN = 'mean'
MEDIAN = 'median'
DELETE = 'delete'
DROP = 'drop'

SALE_PRICE_LOWER_LIMIT = 10000
RENT_PRICE_UPPER_LIMIT = 100000
CONCURRENT_PROCESSES_MAX = 64
SAVE_LABEL_TO_DB = False
DEFAULT_START_DATA_DATE = 20180101
DEFAULT_DATE_POINT_DATE = 20220201


class COLUMN_SUFFIX(Enum):
    NUMBER_CAT = '-c'
    BINARY = '-b'
    NUMBER = '-n'


class Mode(Enum):
    TRAIN = 'train'
    PREDICT = 'predict'
