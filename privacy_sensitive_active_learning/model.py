import datetime
import numpy as np
import pandas as pd
import pickle
import sqlite3
import typing as t

from privacy_sensitive_active_learning import config
from privacy_sensitive_active_learning.data import DataStore
from privacy_sensitive_active_learning.db import sqlite_db
from privacy_sensitive_active_learning.util import init_logging

def warn(*args, **kwargs):
    """
    This will be monkeypatched into the warnings module
    to prevent warnings from being written to console.
    """
    pass

import warnings
warnings.warn = warn

from sklearn.base import clone
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.model_selection import GridSearchCV

conf = config.load()
logger = init_logging(conf['model']['log_level'],__name__)


class Segmentation():
    """
    Keeps track of the 'client side' understanding of the data.
    """

    def __init__(self):
        """
        """

        self.class_probabilities = np.ones(60000, 10) / 10.0


class JobSpec():
    """
    This object is consumed by a fitter worker to tell it what to do.
    In particular, it communicates which generation of the model is being fit, what the class
    being targeted is, and what data is being acted upon.

    ds = DataStore()
    X, y = ds.provide_training_data()
    js = JobSpec(generation=5, digit='d1', ids=)
    """
    def __init__(self, generation: int, target: str, ids: np.ndarray, features: np.ndarray, js_id: int):
        self.generation = generation
        self.digit = target
        self.ids = ids
        self.features = features
        self.js_id = js_id


class NaiveModelProcess():
    """
    An object to manage a single model process's lifetime:    
     + Fit model
     + Score dataset
     + Upload new 'segments' to datastore
    """

    def __init__(self, spec: JobSpec):
        
        self.digit = spec.digit
        self.generation = spec.generation
        self.model = None
        self.X = spec.features
        self.y = spec.ids


    def fit(self):
        """
        Given a desired digit, pull data from the db, fit a linear regression
        and store the updated model for predictions.
        """
        regressor_jobs = conf['model']['regressor_jobs']

        # logger.info("Beginning fit of digit %s...", self.digit)
        start_time = datetime.datetime.now()
        
        param_grid = {'alpha': [0, .1, .01],
                      'l1_ratio': [0.01, .5, 0.99]}

        sgdreg = SGDRegressor(loss='huber', penalty='elasticnet', max_iter=1000, verbose=conf['model']['regressor_verbosity'])

        gscv = GridSearchCV(sgdreg, param_grid, n_jobs=regressor_jobs, cv=3, refit=True, verbose=conf['model']['regressor_verbosity'])

        gscv.fit(self.X, self.y)


        self.model = gscv.best_estimator_

        # RMSE is returned as a negative number, which reflects how GridSearchCV optimizes by maximizing
        # https://stackoverflow.com/questions/21050110/sklearn-gridsearchcv-with-pipeline
        logger.info("digit %s, Generation %s : RMSE: %s", self.digit, self.generation, self.model.score(self.X, self.y))
        elapsed = datetime.datetime.now() - start_time
        logger.info("digit %s, Generation %s : Fit time %s", self.digit, self.generation, elapsed)


    def save(self, filepath: str) -> None:
        """
        Write a model out to a pickle file
        """

        with open(filepath, 'wb') as f:
            pickle.dump(self, f)


    def score(self, X: np.ndarray) -> np.ndarray:
        """
        Score a data matrix
        """
        model = self.model

        return(model.predict(X))


    def return_segment(self, X: np.ndarray, n_devices: int=None, n_samples: int=2) -> t.List[str]:
        """
        Score a feature matrix, select samples from among the highest scoring devices, 
        and for each sample return a list of row IDs.
        """
        n_sample_reinsert = conf['model']['n_sample_reinsert']
        top_n_reinsert = conf['model']['top_n_reinsert']

        if not n_devices:
            n_devices = top_n_reinsert

        logger.debug("digit %s, Generation %s : Size of input X: %s", self.digit, self.generation, X.shape)

        pred_label = self.score(X)

        logger.debug("digit %s, Generation %s : Size of pred_label: %s", self.digit, self.generation, pred_label.shape)

        max_ids = pred_label.argsort()[(-1*n_devices):]

        preds = []

        for _ in range(n_samples):
            # draw a subset of the top IDs for reinsertion
            draws = np.random.choice(max_ids, size=n_sample_reinsert, replace=False)

            id_str = ",".join([str(x) for x in draws.tolist()])
            logger.debug("digit %s, Generation %s : Random ID draws: %s", self.digit, self.generation, id_str)
            logger.info("digit %s, Generation %s : Mean score of selected: %f", self.digit, self.generation, pred_label[draws].mean())
            logger.info("digit %s, Generation %s : Mean score of population: %f", self.digit, self.generation, pred_label.mean())

            preds.append(id_str)

        return preds


class OrdinalModelProcess():
    """
    An object to manage a single model process' lifetime:
     + Fit model
     + Score dataset
     + Produce new 'segments'

    This process will make use of ordinal regression as per
    https://towardsdatascience.com/simple-trick-to-train-an-ordinal-regression-with-any-classifier-6911183d2a3c    
    """

    def __init__(self, spec: JobSpec):

        self.digit = spec.digit
        self.generation = spec.generation
        self.models = {}
        self.unique_class = []
        self.X = spec.features
        self.y = spec.ids
    

    def fit(self):
        """
        This method assumes the data is grouped into ordinal buckets. A series
        of models are fit, where the first predicts Pr(y > cut1), then Pr(y > cut2) 
        and so forth, all implemented as binary classifiers.
        """
        n_ordinal_classes = conf['model']['n_ordinal_classes']

        start_time = datetime.datetime.now()

        # Divide the input space into n classes

        cutpoints = np.linspace(start=0, stop=self.y.max(), num=n_ordinal_classes + 1)
        self.unique_class = [x for x in range(n_ordinal_classes)]

        for m in range(n_ordinal_classes - 1):

            y_class = (self.y > cutpoints[m+1]).astype(np.uint8)

            sgdclf = SGDClassifier(loss='log', penalty='elasticnet')

            class_clf = clone(sgdclf)
            class_clf.fit(self.X, y_class)

            self.models[m] = class_clf

        elapsed = datetime.datetime.now() - start_time
        logger.info("digit %s, Generation %s : Fit time %s", self.digit, self.generation, elapsed)


    def save(self, filepath: str) -> None:
        """
        Write a model out to a pickle file
        """

        with open(filepath, 'wb') as f:
            pickle.dump(self, f)


    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Score a data matrix into its ordinal class
        """

        predicted = []

        classwise_predictions = {k: self.models[k].predict_proba(X) for k in self.models.keys()}
        for i, y in enumerate(self.unique_class):

            if i == 0:
                predicted.append(1-classwise_predictions[i][:, 1])
            elif i in self.models.keys():
                predicted.append(
                    classwise_predictions[i-1][:, 1] - classwise_predictions[i][:, 1]
                )
            else:
                predicted.append(classwise_predictions[i-1][:, 1])

        # vstack produces an n*n_ordinal_classes matrix of probabilities,
        # of which we return the row-wise argmax
        return np.argmax(np.vstack(predicted).T, axis=1)


    def return_segment(self, X: np.ndarray, n_devices: int=None, n_samples: int=2) -> t.List[str]:
        """
        Score a feature matrix, select samples from among the highest scoring devices, 
        and for each sample return a list of row IDs.
        """
        n_sample_reinsert = conf['model']['n_sample_reinsert']
        top_n_reinsert = conf['model']['top_n_reinsert']


        if not n_devices:
            n_devices = top_n_reinsert

        pred_label = self.predict(X)


        # Ideally we'd return segments which only contain our predicted max class.
        # This is not always available (sometimes the max class is not predicted enough 
        # or at all) so we predict the best we can.
        min_acceptable_class = pred_label.max()

        max_ids = np.where(pred_label >= min_acceptable_class)[0]

        while len(max_ids) < 10:
            # row ids in the max class
            min_acceptable_class -= 1
            max_ids = np.where(pred_label >= min_acceptable_class)[0]


        max_ids = np.where(pred_label >= min_acceptable_class)[0]

        logger.debug("max ids: %s", max_ids)

        logger.debug("max_ids shape: %s", max_ids.shape)

        preds = []
        logger.debug("n_sample_reinsert: %s", n_sample_reinsert)

        if n_sample_reinsert > max_ids.shape[0]:
            n_sample_reinsert = int(np.floor(max_ids.shape[0] / 2))

        logger.debug("n_sample_reinsert final: %s", n_sample_reinsert)

        for _ in range(n_samples):
            # draw a subset of the top IDs for reinsertion
            # logger.debug("max_ids: %s, n_sample_reinsert: %s", max_ids, n_sample_reinsert)
            draws = np.random.choice(max_ids, size=n_sample_reinsert, replace=False)

            id_str = ",".join([str(x) for x in draws.tolist()])
            logger.debug("digit %s, Generation %s : Random ID draws: %s", self.digit, self.generation, id_str)
            logger.info("digit %s, Generation %s : Mean score of selected: %f", self.digit, self.generation, pred_label[draws].mean())
            logger.info("digit %s, Generation %s : Mean score of population: %f", self.digit, self.generation, pred_label.mean())

            preds.append(id_str)

        return preds
