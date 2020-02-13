import argparse
import numpy as np
import multiprocessing
import pandas as pd


from privacy_sensitive_active_learning import config
from privacy_sensitive_active_learning.data import DataStore
from privacy_sensitive_active_learning.db import sqlite_db
from privacy_sensitive_active_learning.util import init_logging
from privacy_sensitive_active_learning.process import multi

conf = config.load()

logger = init_logging(conf['log_level'], __name__, root=True)

parser = argparse.ArgumentParser()
parser.add_argument('task', nargs=1, help='which task to run')
args = parser.parse_args()

if args.task[0] == 'naive':
    multi('naive')

if args.task[0] == 'ordinal':
    multi('ordinal')
    