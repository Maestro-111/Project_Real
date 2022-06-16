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
SAVE_LABEL_TO_DB = True


class COLUMN_SUFFIX(Enum):
    NUMBER_CAT = '_c'
    BINARY = '_b'
    NUMBER = '_n'


class Mode(Enum):
    TRAIN = 'train'
    PREDICT = 'predict'
