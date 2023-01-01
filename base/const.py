from enum import Enum


UNKNOWN = 'unknown'
NONE = 'none'
CUSTOM = 'custom'
MEAN = 'mean'
MEDIAN = 'median'
MAX = 'max'
MIN = 'min'
DELETE = 'delete'
DROP = 'drop'

SALE_PRICE_LOWER_LIMIT = 10000
RENT_PRICE_UPPER_LIMIT = 100000
CONCURRENT_PROCESSES_MAX = 1  # 64
SAVE_LABEL_TO_DB = False
DEFAULT_START_DATA_DATE = 20211201
DEFAULT_DATE_POINT_DATE = 20221201


class COLUMN_SUFFIX(Enum):
    NUMBER_CAT = '-c'
    BINARY = '-b'
    NUMBER = '-n'


class Mode(Enum):
    TRAIN = 'train'
    PREDICT = 'predict'
