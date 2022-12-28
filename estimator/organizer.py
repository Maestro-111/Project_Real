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
        self.root_preprocessor = Preprocessor(Mode.TRAIN)
        self.data_source.transform_data(self.root_preprocessor)
        self.__update_status('init transformers', 'done')

    def train_models(self):
        for estimate_manager in [BuiltYearLgbmManager, SqftLgbmEstimateManager, RentLgbmManager, ValueLgbmEstimateManager, SoldPriceLgbmEstimateManager]:
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

    def train_writeback_predictors(self):
        """Train predictors."""
        self.__update_status('train writeback predictors', 'run')
        if self.data_source is None:
            self.init_transformers()
        self.writeback_predictors.clear()
        for Predictor in [BuiltYearEstimator, SqftEstimator, ValueEstimator]:
            predictor = Predictor(
                data_source=self.data_source,
                model_store=self.model_store,
                scale=self.default_sale_scale,
            )
            self.writeback_predictors.append(predictor)
            # later predictor needs the results of previous predictors
            self.__predictor_train_and_writeback(predictor)
        # TODO:
        return
        for Predictor in [RentValueEstimator]:
            predictor = Predictor(
                data_source=self.data_source,
                model_store=self.model_store,
                scale=self.default_rent_scale,
            )
            self.writeback_predictors.append(predictor)
            self.__predictor_train_and_writeback(predictor)
        # self.__train_predictors()
        self.__update_status('train writeback predictors', 'done')

    def __predictor_train_and_writeback(self, predictor):
        score = predictor.train()
        logger.info(f'{predictor.name} score: {score:.4f}')
        if getattr(predictor, 'writeback', None) is not None:
            predictor.writeback()

    def train_predictors(self):
        """Train predictors."""
        self.__update_status('train predictors', 'run')
        if self.data_source is None:
            self.init_transformers()
        self.__build_predictors()
        self.__train_predictors()
        # self.__train_predictors()
        self.__update_status('train predictors', 'done')

    def __build_predictors(self):
        """Build predictors."""
        self.predictors.clear()
        for Predictor in [BuiltYear, Sqft, Value]:
            self.predictors.append(Predictor(
                data_source=self.data_source,
                model_store=self.model_store,
                scale=self.default_scale,
            ))

    def __train_predictors(self):
        """Create predictors and train them."""
        # for predictor in self.predictors:
        #     score = predictor.train()
        #     logger.info(f'{predictor.name} score: {score:.4f}')
        num_procs = min((psutil.cpu_count(logical=False) - 1),
                        CONCURRENT_PROCESSES_MAX,
                        self.num_procs,
                        len(self.predictors))
        logger.info(f'num_procs: {num_procs}')
        timer = Timer(f'predictors', logger)
        pred_results = []
        splitted_predictors = np.array_split(self.predictors, num_procs)
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_procs) as executor:
            results = [
                executor.submit(_predictor_train, predictors)
                for predictors in splitted_predictors
            ]
            for result in concurrent.futures.as_completed(results):
                try:
                    pred_results.append(result.result())
                except Exception as ex:
                    logger.error(str(ex))
                    raise ex
        logger.info(pred_results)
        #self.__update_status('__train predictors', 'done')

    def train(self):
        """Train all the models."""
        self.__update_status('train')
        # 1. extract data from the global data frame
        # 2. clean the data* based on particular estimator and its models
        # 3. split the data into training/test sets
        # 4. create a model
        # 5. train the model
        #    1. when use blending/bagging, need to do more transform on a copied dataset based on particular model requirements.
        # 6. make predictions
        # 7. evaluate and improve
        # 8. save data transform/cleaning and model parameters into database

    def predict_init(self):
        """Initialize the internal structure for prediction."""
        self.__update_status('predict init')
        # 1. load global transformer from db
        # 2. load all models and their transform/cleaning/model parameters from db

    def predict(self):
        """Predict the results."""
        self.__update_status('predict')
        # 1. preliminary transform*
        #    1. when has new labels, raise warning for retraining
        # 2. clean/transform data* based on particular estimator
        # 3. make estimation


def _predictor_train(predictors):
    train_result = []
    for predictor in predictors:
        score = predictor.train()
        # TODO: save models
        train_result.append(f'{predictor.name} : {score}')
    return train_result
