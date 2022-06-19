# top level class for training models and predicting results
# Run 3 modes:
#  1. initialize transformers.
#    - real all data and build label encoder database
#    - save encoder database
#    - hold data in memory
#  2. train models.
#    - load encoder database if not yet
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
import numpy as np
import pandas as pd
import psutil
from base.base_cfg import BaseCfg
from base.const import CONCURRENT_PROCESSES_MAX, Mode
from base.mongo import MongoDB
from base.store_gridfs import GridfsStore
from base.sysDataHelpers import setSysdataTs
from predictor.built_year import BuiltYear
from predictor.sqft import Sqft
from prop.data_source import DataSource
from prop.estimate_scale import EstimateScale, PropertyType
from transformer.preprocessor import Preprocessor
import concurrent.futures
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
        self.predictors = []
        self.default_scale = EstimateScale(
            datePoint=datetime.datetime(2022, 2, 1, 0, 0),
            propType=PropertyType.DETACHED,
            prov='ON',
            area='Toronto',
            city='Toronto',
            sale=True,
        )
        self.num_procs = CONCURRENT_PROCESSES_MAX
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

    def init_transformers(self):
        """Initialize the internal structures for transform and hold data in memory."""
        self.__update_status('init transformers', 'run')
        query = {
            # '_id': {'$in': ['TRBW474049', 'TRBW4874697']},
            'onD': {'$gt': 20200101},
        }
        query = {**query, **self.default_scale.get_query()}
        self.data_source = DataSource(query=query)
        self.root_preprocessor = Preprocessor(Mode.TRAIN)
        self.data_source.load_data(self.root_preprocessor)
        self.__update_status('init transformers', 'done')

    def train_predictors(self):
        """Train predictors."""
        self.__update_status('train predictors', 'run')
        if self.data_source is None:
            self.init_transformers()
        self.__train_predictors()
        self.__update_status('train predictors', 'done')

    def __train_predictors(self):
        """Create predictors and train them."""
        #self.__update_status('__train predictors', 'run')
        self.predictors.clear()
        self.predictors.append(BuiltYear(
            data_source=self.data_source,
            model_store=self.model_store,
            scale=self.default_scale,
        ))
        self.predictors.append(Sqft(
            data_source=self.data_source,
            model_store=self.model_store,
            scale=self.default_scale,
        ))

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
