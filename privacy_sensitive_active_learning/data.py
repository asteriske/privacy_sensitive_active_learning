import numpy as np
import os
import pandas as pd
import pickle
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state 
import sys
import typing as t

from privacy_sensitive_active_learning import config
from privacy_sensitive_active_learning.db import sqlite_db
from privacy_sensitive_active_learning.util import init_logging


conf = config.load()
logger = init_logging(conf['data']['log_level'], __name__)


class DataStore():
    """
    A class that handles data transmission
    """

    def __init__(self, fit_type: str, refresh_db: bool=True):
        
        # Gather MNIST data if necessary
        self.download()
        with open(conf['data']['file'], 'rb') as f:
            data_bunch = pickle.load(f)

        random_state = check_random_state(0)
        permutation = random_state.permutation(data_bunch['data'].shape[0])
        X = data_bunch['data'][permutation, :]
        y = data_bunch['target'][permutation]
        X = X.reshape((X.shape[0], -1))

        self.trainX, self.testX, self.trainY, self.testY = train_test_split(
            X, y, train_size=conf['data']['train_samples'], test_size=10000)

        self.db = sqlite_db(fit_type)
        if refresh_db:
            self.create_db()

        with open(conf['data']['train_test_data_pkl'],'wb') as f:
            pickle.dump({'trainX': self.trainX,
                         'trainY': self.trainY,
                         'testX': self.testX,
                         'testY': self.testY}, f)


    def add_sample(self, sample: dict) -> None:
        """
        Save an oracle response / id group to the labels db
        """

        oracle_summary = sample['oracle_summary']
        id_str = sample['id_str']
        target = sample['target']

        cur = self.db.cursor()

        # logger.debug("ID string: %s" % id_str)
        logger.debug("add_sample oracle_summary: %s", oracle_summary)

        qry = """
            insert into samples (d0,d1,d2,d3,d4,d5,d6,d7,d8,d9,ids,target)
            values (?,?,?,?,?,?,?,?,?,?,?,?)
        """
        cur.execute(qry, tuple(oracle_summary) + tuple([id_str]) + tuple([target]))


    def create_db(self):
        """
        (Re-)create tables to hold data for modeling.
        """

        cur = self.db.cursor()

        cur.execute("drop table if exists samples")

        qry = """
            create table samples (
                d0 float,
                d1 float,
                d2 float,
                d3 float,
                d4 float,
                d5 float,
                d6 float, 
                d7 float,
                d8 float,
                d9 float,
                ids text,
                target char(2),
                ts datetime default current_timestamp
            )
        """

        cur.execute(qry)
        cur.close()

        logger.info("DB 'samples' re-created.")


    def download(self) -> None:
        """
        Test to see if data has been downloaded and stored
        already, if not, download it.
        """

        if not os.path.exists(conf['data']['file']):
            data_bunch = fetch_openml('mnist_784', version=1, return_X_y=False)

            with open(conf['data']['file'], 'wb') as f:
                pickle.dump(data_bunch, f)


    def draw_random_sample(self, n: int=500) -> t.Tuple[np.ndarray, np.ndarray]:
        """
        Draw a single sample of data along with the oracle's breakdown
        """

        ids = np.random.choice(conf['data']['train_samples'], n, replace=True)

        # train_data = self.trainX[ids, :]
        train_labels = self.trainY[ids]

        unique, counts = np.unique(train_labels, return_counts=True)
        oracle_summary = dict(zip(unique, counts / counts.sum()))

        oracle_array = np.zeros((10))

        for k in oracle_summary.keys():
            oracle_array[int(k)] = oracle_summary[k]

        return(ids, oracle_array)


    def draw_oracle_summary(self, ids: np.ndarray) -> np.ndarray:
        """
        Return an oracle summary for submitted row IDs of the training set.
        """
        logger.debug("ids in draw_oracle_summary: %s", ids)
        try:
            assert(max(ids) < 60000), "Invalid ID, out of range"
        except AssertionError:
            sys.exit()

        train_labels = self.trainY[ids]

        unique, counts = np.unique(train_labels, return_counts=True)

        oracle_summary = dict(zip(unique, counts / counts.sum()))

        logger.debug("Oracle pre-summary: %s" % oracle_summary)

        for i in range(10):
            if str(i) not in oracle_summary.keys():
                oracle_summary[str(i)] = 0.0

        logger.debug("Oracle summary: %s:", str(oracle_summary))
        sorted_oracle_summary = {k:oracle_summary[k] for k in sorted(oracle_summary.keys())}
        logger.debug("Sorted oracle summary: %s:", str(sorted_oracle_summary))
        return np.array([x for x in sorted_oracle_summary.values()])


    def provide_training_data(self, digit: str, n_values: int=None) -> t.Tuple[np.ndarray, np.ndarray]:
        """
        For a given digit, query the internal DB for extremely-valued
        data, translate into a matrix and return to the requestor.
        """

        if not n_values:
            n_values = conf['model']['n_top_bottom_values']

        cur = self.db.cursor()

        ## Get top and bottom `n_values` values for the label
        bottom_qry = f"""
            select distinct {digit}, ids
            from samples
            order by {digit}
            limit {n_values}
        """

        bottom_qry_vals = pd.read_sql(bottom_qry, self.db)
        logger.debug("bottom_qry_vals has %d rows", bottom_qry_vals.shape[0])
        logger.debug("data val tail: %s", (bottom_qry_vals.head()))


        middle_qry = f"""
            select distinct {digit}, ids
            from samples
            order by random()
            limit {n_values}
        """

        middle_qry_vals = pd.read_sql(middle_qry, self.db)
        logger.debug("middle_qry_vals has %d rows", middle_qry_vals.shape[0])
        logger.debug("data val tail: %s", (middle_qry_vals.head()))

        top_qry = f"""
            select distinct {digit}, ids
            from samples
            order by {digit} desc
            limit {n_values}
        """

        top_qry_vals = pd.read_sql(top_qry, self.db)
        logger.debug("data val head: %s", (top_qry_vals.head()))
        logger.debug("top_qry_vals has %d rows", top_qry_vals.shape[0])

        # Combine retrieved data frames, extract numpy objects

        # logger.debug("First few ids of top val record: %s" % top_qry_vals['ids'][0].split(',')[0:5])
        # logger.debug("Extreme top labels: %s, %s", top_qry_vals[digit].min(), top_qry_vals[digit].max())
        # logger.debug("Extreme bottom labels: %s, %s", bottom_qry_vals[digit].min(), bottom_qry_vals[digit].max())

        label_array = (
            pd.concat([bottom_qry_vals, middle_qry_vals, top_qry_vals], axis=0)
            .reset_index()
            # Elongate the ids into a column and cast to int
            .assign(id_list=lambda x: x['ids'].str.split(','))
            .explode('id_list')
            .assign(id_list=lambda x: x['id_list'].astype(int))
            [[digit, 'id_list']]

            # extract numpy
            .values
        )

        logger.debug("label_array rows: %d", label_array.shape[0])

        labels = label_array[:, 0]
        ids = label_array[:, 1].astype(int)


        # Grab the relevant rows of the training matrix

        this_training_set = self.trainX[ids, :]

        cur.close()

        return this_training_set, labels
