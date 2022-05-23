# top level class for training models and predicting results

import logging
import socket
import sys
from pandas import DataFrame
from itertools import chain
from base.base_cfg import BaseCfg
from prop.const import Mode
from prop.estimator_built_year import BuiltYear
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from base.mongo import MongoDB
from base.sysDataHelpers import setSysdataTs
from prop.data_reader import read_data_by_query
from transformer.preprocessor import Preprocessor
logger = BaseCfg.getLogger('organizer')


class Organizer:
    """Organizer class for training and predicting models."""

    all_data_query = {
        'onD': {'$gt': 20210901},  # only use data after 2021 for test purpose
        'ptype': 'r',
        'prov': 'ON',
        'area': {
            '$in':
                [
                    'Toronto', 'York', 'Halton', 'Durham',
                    'Peel', 'Hamilton', 'Simcoe',
                ]
        },
    }
    all_data_col_list: list[str] = [
        '_id', 'onD', 'offD', 'lst', 'status',
        'prov', 'area', 'city', 'cmty', 'addr', 'uaddr',
        'st', 'st_num', 'lat', 'lng', 'unt', 'zip',
        'ptype', 'ptype2', 'saletp', 'pstyl', 'ptp',
        'lp', 'lpr', 'sp', 'tax', 'taxyr', 'mfee',
        'bdrms', 'tbdrms', 'br_plus', 'bthrms', 'kch', 'kch_plus',
        'bths', 'rms', 'bsmt', 'schools', 'zone',
        'gr', 'tgr', 'gatp',
        'depth', 'flt',
        'heat', 'feat', 'constr', 'balcony', 'ac',
        'den_fr', 'ens_lndry', 'fce', 'lkr',
        'sqft', 'rmSqft', 'bltYr', 'rmBltYr',
        'cac_inc', 'comel_inc', 'heat_inc', 'prkg_inc',
        'hydro_inc', 'water_inc', 'all_inc',
        'insur_bldg', 'prkg_inc', 'tv',
        'daddr', 'commuId', 'park_fac',
        'comm', 'rltr', 'la', 'la2',
    ]

    def __init__(self):
        """Initialize the class."""
        # connect database / check file system
        self.mongodb = MongoDB()
        self.raw_data: DataFrame = None
        self.prep_data: DataFrame = None
        self.__update_status('init')

    def end(self):
        """Quit the program."""
        self.__update_status('done')
        self.mongodb.close()
        logger.info('Organizer done')

    def run(self):
        """Run the program."""
        self.train_init()
        self.train()
        self.predict_init()
        self.predict()
        self.end()

    def __update_status(self, status):
        self.__status = status
        logger.info('Organizer status: %s', status)
        setSysdataTs(self.mongodb, 'sysdata', {
            'id': f"prop.Organizer_{socket.gethostname()}",
            'batchNm': sys.argv[0],
            'status': status,
        })

    def __init_model(self):
        self.models = []
        # self.models.append((
        #     'built_year', BuiltYear())
        # )

    def train_init(self):
        """Initialize the internal structures for training."""
        # load all data into memory
        self.__update_status('read data')
        self.raw_data = read_data_by_query(
            self.all_data_query, self.all_data_col_list, mongodb=self.mongodb)
        # establish transformer baseline
        self.__update_status('preprocess data')
        self.preprocessor = Preprocessor(Mode.TRAIN)
        # preliminary transformation for reusable columns(e.g. date,labels)
        # self.prep_data = self.preprocessor.fit_transform(self.raw_data)
        # construct estimators and their transformers
        self.__init_model()

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
