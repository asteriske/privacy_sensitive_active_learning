import sqlite3

from privacy_sensitive_active_learning import config 

conf = config.load()


def sqlite_db(fit_type: str):
    """
    Creates and returns a database connection.
    """

    if fit_type == 'naive':
        db_name = 'naive_demosim.db'

    if fit_type == 'ordinal':
        db_name = 'ordinal_demosim.db'

    return sqlite3.connect(db_name)
