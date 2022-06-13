

from datetime import datetime
from enum import Enum
import pandas as pd
from base.base_cfg import BaseCfg
from base.timer import Timer
from prop.data_source import DataSource
from prop.estimate_scale import EstimateScale
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from base.util import print_dateframe
from sklearn.metrics import explained_variance_score, accuracy_score, r2_score

from base.store_file import FileStore


class ModelClass(Enum):
    Regression = 'Regression'
    Classification = 'Classification'


class BasePredictor(BaseEstimator):
    """Predictor class

    Parameters
    ==========
    x_columns: list[str] collection of columns to be used as features
    x_required: list[str] collection of required columns. If not specified, x_columns is used.

    """

    def __init__(
        self,
        name: str = 'BasePredictor',
        data_source: DataSource = None,
        scale: EstimateScale = None,
        model=None,
        model_params: dict = None,
        col_list: list[str] = None,
        x_columns: list[str] = None,
        x_required: list[str] = None,
        y_column: str = None,
        y_numeric_column: str = None,
        source_filter_func: (pd.Series) = None,
        source_date_span: int = None,
        model_store=None,
        model_class: ModelClass = ModelClass.Regression,
    ) -> None:
        self.data_source = data_source
        self.name = name if name else self.__class__.__name__
        self.model = model
        self.model_params = model_params
        self.model_class = model_class
        self.scale = scale
        self.col_list = col_list
        self.x_columns = x_columns
        self.x_required = x_required if x_required else x_columns
        self.y_column = y_column
        self.y_numeric_column = y_numeric_column if y_numeric_column else y_column
        self.col_list = self.x_columns.copy()
        self.col_list.append(self.y_column)
        self.source_filter_func = source_filter_func
        self.source_date_span = source_date_span
        if model_store is None:
            self.store = FileStore()
        else:
            self.store = model_store

        self.pred_accuracy_score = None
        self.data: pd.DataFrame = None
        self.data_train = None
        self.trained_seconds = 0
        self.predict_per_second = 0
        self.logger = BaseCfg.getLogger(self.name)

    def load_model(self):
        """Load the model"""
        data, accuracy, meta = self.store.load_data(self.name)
        self.pred_accuracy_score = accuracy
        self.model = data
        for k, v in meta.items():
            setattr(self, k, v)
        return self.model

    def save_model(self, model=None):
        """Save the model"""
        if model is None:
            model = self.model
        if model is None:
            raise ValueError('Model is not specified.')
        self.model_path = self.store.save_data(
            self.name,
            model,
            self.pred_accuracy_score,
            self.get_meta(),
        )
        return self.model_path

    def get_meta(self) -> dict:
        return {
            'name': self.name,
            'accuracy_score': self.pred_accuracy_score,
            'created_ts': self.created_ts,
            'trained_seconds': self.trained_seconds,
            'predict_per_second': self.predict_per_second,
            'model_params': self.model_params,
        }

    # TODO: implement the following methods
    def set_params(self, scale: EstimateScale, col_list: list[str],):
        self.scale = scale

    def set_scale(self, scale: EstimateScale):
        """Set the scale"""
        self.scale = scale
        # need to reload the data
        self.data = None
        self.data_train = None

    def load_data(
        self,
        date_span: int = None,
        filter_func: (pd.Series) = None,
    ) -> pd.DataFrame:
        """Load the data"""
        if self.data_source is None:
            raise ValueError('Data source is not specified.')
        self.data = self.data_source.get_df(
            scale=self.scale,
            cols=self.col_list,
            date_span=date_span or self.source_date_span,
            filter_func=filter_func or self.source_filter_func,
        )
        self.generate_numeric_columns()
        return self.data

    def prepare_data(self, X, params=None):
        """Prepare the data for training or prediction. 
        To be implemented by subclasses.
        """
        return X

    def prepare_model(self):
        """Prepare the model.
        To be implemented by subclasses.
        """
        self.created_ts = datetime.now()

    def generate_numeric_columns(self):
        """Return the number of columns"""
        self.x_numeric_columns_ = []
        for col in self.data.columns:
            if self.data[col].dtype == 'float64' or self.data[col].dtype == 'int64':
                self.x_numeric_columns_.append(col)
        return self.x_numeric_columns_

    def train(self):
        """Train the model"""
        if self.data_train is None:
            if self.data is None:
                self.load_data()
            self.data_train = self.prepare_data(self.data)
        df_train, df_test = train_test_split(
            self.data_train, test_size=0.15, random_state=10)
        # X = self.data_train[self.x_columns].values
        # y = self.data_train[self.y_column].values
        # X_train, X_test, y_train, y_test = train_test_split(
        #     X, y, random_state=0)
        timer = Timer(self.name, self.logger)
        self._train(df_train[self.x_numeric_columns_],
                    df_train[self.y_numeric_column])
        self.pred_accuracy_score = self.test(
            X=None,
            X_test=df_test[self.x_numeric_columns_],
            y=df_test[self.y_numeric_column])
        self.trained_seconds = timer.stop(df_train.shape[0])
        # print(f'{self.name} accuracy: {self.pred_accuracy_score}')
        return self.pred_accuracy_score

    def _train(self, X, y, **kwargs):
        """Train the model."""
        self.prepare_model()
        self.fit(X, y, **kwargs)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit the model"""
        self.model.fit(X, y)
        pass

    def test(self, X, y, X_test=None):
        """Calculate the accuracy of the model."""
        if X_test is None:
            X_test = self.prepare_data(X)
        y_pred = self.predict(X_test)
        self.pred_accuracy_score = round(self.get_score(y, y_pred) * 100)
        return self.pred_accuracy_score

    def get_score(self, y_true, y_pred):
        """Get the accuracy score."""
        if self.model_class == ModelClass.Regression:
            pred_accuracy_score = r2_score(y_true, y_pred)
        elif self.model_class == ModelClass.Classification:
            pred_accuracy_score = accuracy_score(y_true, y_pred)
        else:
            raise ValueError('Model class is not specified.')
        return pred_accuracy_score

    def tune(self, date_spans: list[int] = None):
        """Tune the model. Two levels of tuning: 1. date_span 2. model_params"""
        if date_spans is None:
            # 6months, 1year, 2years, 4years, 8years
            date_spans = [180, 365, 730, 1461, 2922]
        best_date_span = 0
        best_accuracy_score = 0
        for date_span in date_spans:
            self.data_train = None
            self.load_data(date_span)
            # TODO use different model_params
            self.train()
            if self.pred_accuracy_score > best_accuracy_score:
                best_date_span = date_span
                best_accuracy_score = self.pred_accuracy_score
            # self.save_model()
        self.logger.info(
            f'{self.name} best date_span: {best_date_span} accuracy: {best_accuracy_score}', self.scale)
        return best_date_span, best_accuracy_score

    def tune_model(self, X, y, params, data_params=None):
        """Tune the model."""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, random_state=0)
        X_train = self.prepare_data(X_train, data_params)
        X_test = self.prepare_data(X_test, data_params)
        scores = [0] * len(params)
        for i in range(len(params)):
            self.model_param = params[i]
            self.train(None, y_train, X_train=X_train)
            scores[i] = self.test(None, y_test, X_test=X_test)
        max_accuracy_index = scores.index(max(scores))
        max_accuracy_params = params[max_accuracy_index]
        self.model_param = params
        return (scores, max_accuracy_index, max_accuracy_params)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Predict the results"""
        return self.model.predict(X)

    def __str__(self) -> str:
        return self.__class__.__name__
