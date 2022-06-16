# Mapping the labels to the corresponding numbers.
# Save the mapping to database and reuse when needed.
from base.base_cfg import BaseCfg
from base.mongo import getMongoClient
from base.timer import Timer
from base.const import SAVE_LABEL_TO_DB, Mode
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd

logger = BaseCfg.getLogger(__name__)


class DbLabelTransformer(TransformerMixin, BaseEstimator):
    """Transforms labels to integers and save the mapping to database

    Parameters
    ----------
    collection : mongodb collection
        The mongodb collection to save the mapping to.
    col: str
        The feature name to save the mapping as.
    mode: Mode
        The mode of the transformer. PREDICT or TRAIN.
    na_value: str
        The value to fill the empty fields with.

    Attributes
    ----------
    n_features_ : int
        The number of features of the data passed to :meth:`fit`.
        Right now, this is always 1.
    labels_ : dict
        A dictionary mapping the column label names to the label indexes.
    labels_index_ : dict
        A dictionary mapping the column label indexes to the label names.
    labels_index_next_ : int
        The next label index to use.
    """

    def __init__(
        self,
        collection,
        col: str,
        mode: Mode = Mode.TRAIN,
        na_value=None,
        save_to_db: bool = SAVE_LABEL_TO_DB,
    ):
        if collection is None:
            return
        self.collection = collection
        self.col = col
        self.mode = mode
        self.na_value = na_value
        self.save_to_db = save_to_db
        # create index on col
        MongoDB = getMongoClient()
        MongoDB.createIndex(self.collection, fields=[
            ("col", 1), ('i', 1)], unique=True)
        # load from database when predicting
        if self.mode == Mode.PREDICT:
            self.labels_ = {}
            self.labels_index_ = {}
            self.labels_index_next_ = 0
            for doc in MongoDB.find(self.collection, {'col': self.col}):
                self.labels_[doc['col']][doc['label']] = doc['i']
                self.labels_index_[doc['col']][doc['i']] = doc['label']
                if doc['i'] > self.labels_index_next_[self.col]:
                    self.labels_index_next_[self.col] = doc['i']

    def _save_mapping(self, col: str, label: str, index: int, count: int):
        if not self.save_to_db:
            return
        getMongoClient().save(self.collection, {
            "_id": f"{col}:{label}",
            "col": col,
            "label": label,
            "i": index,
            "cnt": count,
        })

    def fit(self, X, y=None):
        """A reference implementation of a fitting function for a transformer.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.
        Returns
        -------
        self : object
            Returns self.
        """
        logger.debug(f'label {self.col} fit {self.mode}')
        # X = check_array(X, accept_sparse=True)

        # self.n_features_ = X.shape[1]

        if self.mode == Mode.TRAIN:
            t = Timer(self.col, logger)
            t.start()
            col = self.col  # TODO: deal with multiple columns
            getMongoClient().deleteMany(self.collection, {'col': col})
            self.labels_ = {col: {}}
            self.labels_index_ = {col: {}}
            self.labels_index_next_ = {col: 0}
            col_labels = X.value_counts()
            col_labels.index = col_labels.index.astype(str)
            col_labels = col_labels.sort_index()
            totalLabels = 0
            for label, count in col_labels.items():
                if label is None:
                    if self.na_value is not None:
                        label = self.na_value
                    else:
                        continue
                cur_index = self.labels_index_next_[col]
                self.labels_[col][label] = cur_index
                self.labels_index_[col][cur_index] = label
                self._save_mapping(col, label, cur_index, count)
                self.labels_index_next_[col] += 1
                totalLabels += 1
            # print(self.labels_)
            logger.info(
                f'{self.col} fit {self.mode} labels:{totalLabels}')
            t.stop(X.shape[0])

        # Return the transformer
        return self

    def transform(self, X):
        """ A reference implementation of a transform function.
        Parameters
        ----------
        X : {array-like, sparse-matrix}, shape (n_samples, n_features)
            The input samples.
        Returns
        -------
        X_transformed : array, shape (n_samples, n_features)
            The array containing the element-wise square roots of the values
            in ``X``.
        """
        # Check is fit had been called
        # check_is_fitted(self, 'n_features_')

        # # Input validation
        # X = check_array(X, accept_sparse=True)

        # # Check that the input is of the same shape as the one passed
        # # during fit.
        # if X.shape[1] != self.n_features_:
        #     raise ValueError('Shape of input is different from what was seen'
        #                      'in `fit`')
        # we may need to use a wrapper here
        # X = pd.DataFrame(X)
        logger.debug(f'label {self.col} transform {self.mode}')
        if self.mode == Mode.TRAIN:
            if (getattr(self, 'labels_', None) is None) or (self.col not in self.labels_):
                self.fit(X)
        # fill na_value
        if self.na_value is not None:
            X.fillna(value=self.na_value, inplace=True)

        # transform with new column names and add new labels as needed
        X = X.apply(str)  # convert string
        mapping = self.labels_[self.col]

        def map_value(label, mapping, self, col):
            index = mapping.get(label)
            if index is None:
                logger.info(f'unknown label {label} in column {col}')
                index = self.labels_index_next_[col]
                mapping[label] = index
                self.labels_index_next_[col] += 1
                self._save_mapping(col, label, index, 0)
            return index
        X = X.apply(
            lambda val: map_value(val, mapping, self, self.col))
        # return X
        return X
