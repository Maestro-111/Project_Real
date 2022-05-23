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


class Mode(Enum):
    TRAIN = 'train'
    PREDICT = 'predict'
