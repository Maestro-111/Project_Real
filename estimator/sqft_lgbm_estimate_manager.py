from data.estimate_scale import EstimateScale
from estimator.lgbm_estimate_manager import LgbmEstimateManager
import pandas as pd
from math import isnan

from estimator.writeback_mixin import WritebackMixin

ESTIMATOR_NAME = 'Sqft'


def needSqft(row):
    return not (row['sqft-n'] is None or isnan(row['sqft-n']))


class SqftLgbmEstimateManager(LgbmEstimateManager, WritebackMixin):
    """Sqft estimate manager."""

    y_db_col = 'sqft_m1'
    y_column = 'sqft'
    x_columns = [
        'lat', 'lng',
        'bthrms', 'pstyl',
        'rms', 'bths',
        'bdrms', 'tbdrms', 'br_plus', 'bthrms', 'kch', 'kch_plus',
        'zip',  'gatp',
        'depth', 'flt',
        'st_num-st-n',
        'ptp', 'pstyl', 'constr', 'feat',
        'bsmt',  'heat', 'park_fac',
        'st', 'st_num',
    ]

    def __init__(self, data_source):
        super().__init__(
            data_source,
            name=ESTIMATOR_NAME,
            model_params=None,
        )

    def my_load_data(self, scale: EstimateScale = None) -> pd.DataFrame:
        return self.load_data(
            scale=scale,
            filter_func=needSqft,
        )
