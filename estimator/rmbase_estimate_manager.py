from datetime import datetime
from enum import Enum
from base.base_cfg import BaseCfg
from base.model_store import ModelStore
from data.data_source import DataSource
import pandas as pd
from sklearn.metrics import explained_variance_score, accuracy_score, r2_score

from data.estimate_scale import EstimateScale


class ModelClass(Enum):
    Regression = 'Regression'
    Classification = 'Classification'


class RmBaseEstimateManager:
    """RM base estimate manager.
    Each estimate manager is responsible for one type of estimate,
    and shall extend this class.

    attributes:
        x_columns: list[str] 
        y_column: str
        date_span: int
        filter_func: pd.Series -> bool
        suffix_list: list[str]
        scales: {repr(scale): EstimateScale}
            model, meta, accuracy are stored in each scale.
        scale: EstimateScale
            when there is only one scale, this is the scale.
            either scale or scales is set.

    generated attributes:
        df: pd.DataFrame


    Parameters
    ==========
    DataSource: data.data_source.DataSource
    """

    def __init__(
        self,
        data_source: DataSource,
        name: str = None,
        model_class: ModelClass = ModelClass.Regression,
    ) -> None:
        self.data_source = data_source
        self.name = name
        self.model_class = model_class
        self.logger = BaseCfg.getLogger(name or self.__class__.__name__)
        pass

    def load_scales(self, sale: bool = None) -> None:
        """Load the scales or scale suitable for this estimator.
        This step must be done before train/save/load.
        To support other filters, override this method.
        """
        self.scales = {}
        for scale in self.data_source.scale.getLeafScales(sale=sale):
            self.scales[repr(scale)] = scale

    def train(self) -> None:
        """Train the estimator.
        Train the estimator(s) for the specified scale or all scales.
        """
        if hasattr(self, 'scale'):
            scale, model, accuracy, x_cols, meta = self.train_single_scale(
                scale)
            scale.model = model
            scale.accuracy = accuracy
            scale.x_cols = x_cols
            scale.meta = meta
        elif hasattr(self, 'scales'):
            for scale in self.scales.values():
                scale, model, accuracy, x_cols, meta = self.train_single_scale(
                    scale)
                scale.meta['model'] = model
                scale.meta['accuracy'] = accuracy
                scale.meta['x_cols'] = x_cols
                scale.meta['meta'] = meta
        else:
            self.logger.warning(
                'No scale or scales defined. Load scales first.')
            self.load_scales()
            self.train()

    def train_single_scale(self, scale: EstimateScale) -> tuple[EstimateScale, object, float, list[str], dict]:
        """Train the estimator for a single scale.
        This method is called by train method.
        """
        raise NotImplementedError

    def estimate(self, X: pd.DataFrame, scale: EstimateScale = None) -> pd.DataFrame:
        """Estimate the data source.
        Either train or load must be called before this.
        steps:
            1. group data by scale if not specified
            2. preprocess data
            3. estimate by scale estimator
            4. merge results
        """
        # TODO: implement this
        raise NotImplementedError

    def estimate_by_id(self, id: str) -> pd.DataFrame:
        """Estimate the data source by id.
        Either train or load must be called before this.
        steps:
            1. read data from database
            2. use estimate method to estimate
        """
        # TODO: implement this
        raise NotImplementedError

    def save(self, store: ModelStore) -> None:
        """Save the estimator(s).
        Shall save all the estimators. train or load must be called before this.
        """
        if hasattr(self, 'scale'):
            self.save_one_model(
                store, (self.scale, self.model, self.accuracy, self.meta))
        elif hasattr(self, 'scales'):
            for scale in self.scales.values():
                data = (scale, scale.meta['model'],
                        scale.meta['accuracy'], scale.meta['meta'])
                self.save_one_model(store, data)
        else:
            raise Exception('No scale or scales is set.')

    def save_one_model(self, store: ModelStore, data: tuple[EstimateScale, any, float, dict]) -> None:
        """Save one estimator."""
        # estimator name, model name, scale, date
        filename = ':'.join([self.name, self.model_name, repr(data[0])])
        meta = {
            'accuracy': data[2],
            'ts': datetime.now(),
        }
        if len(data) > 3:
            meta.update(data[3])
        self.logger.info('Saving model: %s', filename)
        store.save_model(filename, data[1], data[2], meta)

    def load(self, store: ModelStore):
        """Load the estimator(s)."""
        if hasattr(self, 'scale'):
            scale, self.model, self.accuracy, self.meta = self.load_one_model(
                store, self.scale)
        elif hasattr(self, 'scales'):
            for scale in self.scales.values():
                scale, scale.meta['model'], scale.meta['accuracy'], scale.meta['meta'] = self.load_one_model(
                    store, scale)
        else:
            raise Exception('No scale or scales is set.')

    def load_one_model(self, store: ModelStore, scale: EstimateScale) -> tuple[EstimateScale, any, float, dict]:
        """Load one estimator."""
        filename = ':'.join([self.name, self.model_name, repr(scale)])
        try:
            model, accuracy, meta = store.load_model(filename)
        except FileNotFoundError as e:
            self.logger.error('Can not find model: %s', filename)
            raise e
        except Exception as e:
            self.logger.error('Failed to load model: %s', filename)
            raise e
        return (scale, model, accuracy, meta)

    def propToScale(self, prop: str) -> EstimateScale:
        """Get the scale from the property."""
        # TODO: implement this
        return self.scales[prop]

    # ---- Feature selection ----
    def select_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Select features."""
        raise NotImplementedError

    def tune(self) -> None:
        """Tune the estimator."""
        raise NotImplementedError

    def test(self) -> None:
        """Test the estimator."""
        raise NotImplementedError

    # ---- Supporting functions ----
    def load_data(
        self,
        scale: EstimateScale,
        x_columns: list[str] = None,
        y_column: str = None,
        date_span: int = None,
        filter_func: (pd.Series) = None,
        suffix_list: list[str] = None,
        numeric_columns_only: bool = None,
        prefer_estimated: bool = False,
    ) -> pd.DataFrame:
        """Load the data and set it to self.df.
        Use the default values if not specified:
        self.x_columns, self.y_column, self.date_span, 
        self.filter_func, self.suffix_list
        """
        if self.data_source is None:
            raise ValueError('Data source is not specified.')
        col_list = []
        if x_columns is not None:
            col_list.extend(x_columns)
        elif hasattr(self, 'x_columns'):
            self.logger.info(f'x_columns: {self.x_columns}')
            col_list.extend(self.x_columns)
        if len(col_list) == 0:
            # when no x_columns are specified, load all columns
            col_list = None
        elif y_column is not None:
            col_list.append(y_column)
        elif hasattr(self, 'y_column'):
            col_list.append(self.y_column)
        numeric_columns_only = numeric_columns_only or getattr(
            self, 'numeric_columns_only', False)
        # do not save df to self.df, as subclases may need to load different data sets multiple times
        df = self.data_source.get_df(
            scale=scale,
            cols=col_list,
            date_span=date_span or self.date_span,
            filter_func=filter_func,  # self.filter_func: bound method does not work
            suffix_list=suffix_list or self.suffix_list,
            numeric_columns_only=numeric_columns_only,
            prefer_estimated=prefer_estimated,
        )
        self.logger.info(f'Data loaded. {suffix_list or self.suffix_list}')
        return df

    def filter_data(
        self,
        X,
    ):
        """ Filter data. To be implemented by subclasses."""
        return X

    def get_x_y_columns(
        self,
        df: pd.DataFrame,
        y_column: str = None,
    ) -> tuple[list[str], str]:
        """Get numeric columns of x and y from df."""
        df = self.filter_data(df)
        y_column = y_column or self.y_column
        y_numeric_column = None
        x_numeric_columns = []
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                x_numeric_columns.append(col)
                if col == y_column:
                    y_numeric_column = col
                elif col.startswith(y_column) and col[-2:] in ['-b', '-n', '-c']:
                    y_numeric_column = col
        if y_numeric_column is None:
            raise ValueError(f'Column {y_column} is not numeric.')
        x_numeric_columns.remove(y_numeric_column)
        self.logger.info(
            f'*{self.name}* X: {x_numeric_columns} y: {y_numeric_column}')
        return x_numeric_columns, y_numeric_column

    def test_accuracy(self, model, X_test, y_test) -> float:
        """Calculate the accuracy of the model."""
        y_pred = model.predict(X_test)
        self.pred_accuracy_score = round(
            self.get_score(y_test, y_pred) * 10000)
        return self.pred_accuracy_score

    def get_score(self, y_true, y_pred) -> float:
        """Get the accuracy score."""
        if self.model_class is None or self.model_class == ModelClass.Regression:
            pred_accuracy_score = r2_score(y_true, y_pred)
        elif self.model_class == ModelClass.Classification:
            pred_accuracy_score = accuracy_score(y_true, y_pred)
        else:
            raise ValueError(
                f'Model class is not specified. {self.model_class}')
        return pred_accuracy_score
