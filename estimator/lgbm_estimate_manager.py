from base.timer import Timer
from data.estimate_scale import EstimateScale
from estimator.rmbase_estimate_manager import ModelClass, RmBaseEstimateManager
import lightgbm as lgb
import pandas as pd
from math import isnan
from sklearn.model_selection import train_test_split, RepeatedKFold, cross_val_score


class LgbmEstimateManager(RmBaseEstimateManager):
    """LightGBM manager."""

    model_name = 'lgbm'
    date_span = 365
    suffix_list = ['-n', '-c', '-b', '-e']
    numeric_columns_only = True
    default_model_params = {
        'n_estimators': 300,
        'max_depth': -1,
        'num_leaves': 100,
    }

    def __init__(
        self,
        data_source,
        name,
        model_params,
    ):
        super().__init__(
            data_source,
            name,
            model_class=ModelClass.Regression,
        )
        self.model_params = model_params

    def prepare_model(self):
        """Prepare model."""
        model_params = self.model_params or self.default_model_params
        self.model = lgb.LGBMRegressor(**model_params)
        self.logger.info('model_params: {model_params}')
        return self.model

    def filter_data(
        self,
        X,
    ):
        """ Filter data for LGBMRegressor """
        # X = X.copy()
        rowsBefore = X.shape[0]
        colsBefore = X.shape[1]
        # remove columns with all NaN
        X.dropna(axis='columns', how='all', inplace=True)
        # remove columns with all zeros
        X = X.loc[:, (X != 0).any(axis=0)]
        # remove rows with NaN
        X.dropna(inplace=True)
        rowsAfter = len(X.index)
        colsAfter = X.shape[1]
        self.logger.info(
            f'''*{self.name}* Rows dropped: {rowsBefore-rowsAfter}/{rowsBefore}=>{rowsAfter}
            Cols dropped: {colsBefore-colsAfter}/{colsBefore}=>{colsAfter}''')
        return X

    def train_single_scale(self, scale: EstimateScale) -> tuple[EstimateScale, object, float, list[str], dict]:
        timer = Timer(str(scale), self.logger)
        timer.start()
        df = self.my_load_data(scale)
        df_train, df_test = train_test_split(
            df, test_size=0.15, random_state=10)
        model = self.prepare_model()
        x_cols, y_col = self.get_x_y_columns(df)
        model.fit(df_train[x_cols], df_train[y_col])
        accuracy = self.test_accuracy(
            model, df_test[x_cols], df_test[y_col])
        timer.stop(df_train.shape[0])
        self.logger.info('================================================')
        self.logger.info(
            f'{str(scale)} {str(self.model_name)} model trained accuracy:{accuracy/100.0}%')
        featureImportance = self.feature_importance(model)
        self.logger.info('------------------------------------------------')
        featureImportanceDic = {}
        for weight, feature in featureImportance:
            featureImportanceDic[feature] = int(weight)
            self.logger.info(f'{feature}: {weight}')
        return (scale, model, accuracy, x_cols, {'featureImportance': featureImportanceDic})

    def feature_importance(self, model) -> list:
        featureZip = list(zip(model.feature_importances_, model.feature_name_))
        featureZip.sort(key=lambda v: v[0], reverse=True)
        return featureZip

    def my_load_data(self, scale: EstimateScale = None) -> pd.DataFrame:
        """Subclass can override this method to load data.

        Args:
            scale (EstimateScale, optional): Defaults to None.

        Returns:
            pd.DataFrame: dataframe
        """
        return self.data_source.load_data(scale)
