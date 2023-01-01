# top level class for training models and predicting results
# Run 3 modes:
#  1. initialize transformers.
#    - real all data and build label encoder database
#    - save encoder database
#    - hold data in memory
#  2. train models.
#    - load encoder database if not yet
#    - train and writeback predictors
#    - train models
#    - save models
#  3. predict results.
#    - load encoder database if not yet
#    - load models
#    - predict results

import datetime
import socket
import sys
from pandas import DataFrame
from base.timer import Timer
from base.util import dateFromNum
import numpy as np
import pandas as pd
from predictor.rent_value import RentValue, RentValueEstimator
from predictor.value import Value, ValueEstimator
from predictor.writeback_mixin import WriteBackMixin
import psutil
from base.base_cfg import BaseCfg
from base.const import CONCURRENT_PROCESSES_MAX, DEFAULT_DATE_POINT_DATE, DEFAULT_START_DATA_DATE, Mode
from base.mongo import MongoDB
from base.model_gridfs_store import GridfsStore
from base.sysDataHelpers import setSysdataTs
from predictor.built_year import BuiltYear, BuiltYearEstimator
from predictor.sqft import Sqft, SqftEstimator
from data.data_source import DataSource
from data.estimate_scale import EstimateScale, PropertyType
from transformer.preprocessor import Preprocessor
import concurrent.futures

from estimator.builtyear_lgbm_estimate_manager import BuiltYearLgbmManager
from estimator.sqft_lgbm_estimate_manager import SqftLgbmEstimateManager
from estimator.rent_lgbm_estimate_manager import RentLgbmManager
from estimator.value_lgbm_estimate_manager import ValueLgbmEstimateManager
from estimator.soldprice_lgbm_estimate_manager import SoldPriceLgbmEstimateManager


logger = BaseCfg.getLogger(__name__)


class Organizer:
    """Organizer class for training and predicting models."""

    def __init__(self):
        """Initialize the class."""
        # connect database / check file system
        self.mongodb = MongoDB()
        self.raw_data: DataFrame = None
        self.prep_data: DataFrame = None
        self.data_source: DataSource = None
        self.model_store = GridfsStore(
            # self.mongodb.getDb(),
            collection='ml_fs',
            prefix='ml_',
        )
        self.root_preprocessor: Preprocessor = None
        self.writeback_predictors = []
        self.predictors = []
        self.default_all_scale = EstimateScale(
            datePoint=dateFromNum(DEFAULT_DATE_POINT_DATE),
            propType=None,  # PropertyType.DETACHED,
            prov='ON',
            area='Peel',
            city='Mississauga',
            sale=None,
        )
        self.default_sale_scale = EstimateScale(
            datePoint=dateFromNum(DEFAULT_DATE_POINT_DATE),
            propType=PropertyType.SEMI_DETACHED,
            prov='ON',
            area='Toronto',
            city='Toronto',
            sale=True,
        )
        self.default_rent_scale = EstimateScale(
            datePoint=dateFromNum(DEFAULT_DATE_POINT_DATE),
            propType=PropertyType.DETACHED,
            prov='ON',
            area='Toronto',
            city='Toronto',
            sale=False,
        )
        self.num_procs = CONCURRENT_PROCESSES_MAX
        self.estimate_managers = {}
        # self.similar_properties = {'on': {}, 'off': {}}
        # self.timed_values = {'sale':{}, 'rent':{}}
        self.__update_status('program', 'run')

    def __update_status(self, job, status):
        self.__status = status
        logger.info(f'Organizer job({job}) status: {status}')
        setSysdataTs(self.mongodb, 'sysdata', {
            'id': f"prop.Organizer_{socket.gethostname()}",
            'batchNm': sys.argv[0],
            'job': job,
            'status': status,
            'mt': BaseCfg.getTs()
        })

    def end(self):
        """Quit the program."""
        self.__update_status('program', 'done')
        self.mongodb.close()
        logger.info('Organizer end')

    def load_data(self):
        """Load data from database."""
        self.__update_status('load data', 'run')
        query = {
            # '_id': {'$in': ['TRBW474049', 'TRBW4874697']},
            'onD': {'$gt': DEFAULT_START_DATA_DATE},
        }
        self.data_source = DataSource(
            scale=self.default_all_scale,
            query=query)
        self.data_source.load_raw_data()
        self.__update_status('load data', 'done')

    def init_transformers(self):
        """Initialize the internal structures for transform and hold data in memory."""
        self.__update_status('init transformers', 'run')
        if self.data_source is None:
            self.load_data()
        self.root_preprocessor = Preprocessor()
        self.data_source.transform_data(self.root_preprocessor)
        self.__update_status('init transformers', 'done')

    def train_models(self):
        for estimate_manager in [
            BuiltYearLgbmManager,
            SqftLgbmEstimateManager,
            RentLgbmManager,
            ValueLgbmEstimateManager,
            SoldPriceLgbmEstimateManager,
        ]:
            em = estimate_manager(self.data_source)
            em.load_scales()
            em.train()
            if em.name not in self.estimate_managers:
                self.estimate_managers[em.name] = {}
            self.estimate_managers[em.name][em.model_name] = em
            logger.info(f"trained {em.name} {em.model_name}")
            if hasattr(em, 'writeback'):
                em.writeback()
                logger.info(f"writeback {em.name} {em.model_name}")

    def save_models(self):
        for estimate_manager in self.estimate_managers.values():
            for em in estimate_manager.values():
                em.save(self.model_store)

    def load_models(self):
        """load models from database."""
        # TODO: load models from database
        pass

    def predict(self, id_list: list[str], writeback: bool = False):
        """Predict the values for the given list of ids."""
        self.__update_status('predict', 'run')
        if self.data_source is None:
            self.load_data()
        if self.root_preprocessor is None:
            self.init_transformers()
        if not self.estimate_managers:
            self.load_models()
        df_grouped_result, added_cols = self.__predict(id_list, writeback)
        # self.__predict_parallel(id_list)
        self.__update_status('predict', 'done')
        return df_grouped_result, added_cols

    def __predict(
        self,
        id_list: list[str],
        writeback: bool = False
    ) -> tuple[pd.DataFrame, list[str]]:
        """Predict the values for the given list of ids."""
        df_grouped = self.data_source.load_df_grouped(
            id_list, self.root_preprocessor)
        added_cols = []
        for estimate_manager in self.estimate_managers.values():
            # estimater type lavel
            for em in estimate_manager.values():
                # model level
                logger.info(
                    f'-------------predict {em.name} {em.model_name} ----------------------------')
                df_y, y_col_names = em.estimate(df_grouped)
                if df_y is None:
                    continue
                logger.info(
                    f"predict {em.name} {em.model_name} {y_col_names}:")
                logger.info(df_y)
                logger.info('-----------------------------------------')
                # merge the prediction to the original data
                # df_grouped = pd.merge(
                #     df_grouped, df_y, how='left', left_index=True, right_index=True)
                df_grouped = self.data_source.writeback(
                    y_col_names, df_y, df_grouped)
                added_cols.extend(y_col_names)
        return df_grouped, added_cols
