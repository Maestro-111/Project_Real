from data.estimate_scale import EstimateScale
from estimator.lgbm_estimate_manager import LgbmEstimateManager
import pandas as pd
from math import isnan

from estimator.writeback_mixin import WritebackMixin

ESTIMATOR_NAME = 'Value'


def needSold(row):
    return (not (row['sp-n'] is None or isnan(row['sp-n']))) and \
        (row['sp-n'] > 200000)


class ValueLgbmEstimateManager(LgbmEstimateManager, WritebackMixin):
    """Value estimate manager."""

    y_target_col = 'value-n-e'
    y_column = 'sp-n'
    x_columns = [
        'onD',
        'cmty',
        'st', 'st_num', 'lat', 'lng',
        'pstyl', 'ptp',
        'tax', 'taxyr',
        'bdrms', 'tbdrms', 'br_plus', 'bthrms', 'kch', 'kch_plus',
        'bths', 'rms', 'bsmt', 'schools', 'zone',
        'gr', 'tgr', 'gatp',
        'depth', 'flt',
        'heat', 'feat', 'constr', 'balcony', 'ac',
        'den_fr', 'ens_lndry', 'fce',
        'sqft', 'bltYr',
        'park_fac',
    ]

    def __init__(self, data_source):
        super().__init__(
            data_source,
            name=ESTIMATOR_NAME,
            model_params=None,
        )

    def load_scales(self, sale: bool = None) -> None:
        return super().load_scales(sale=True)

    def my_load_data(self, scale: EstimateScale) -> pd.DataFrame:
        scale = scale.copy(sale=True)
        return self.load_data(
            scale=scale,
            filter_func=needSold,
        )
