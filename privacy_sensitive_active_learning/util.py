import logging
import sys

from privacy_sensitive_active_learning import config

conf = config.load()

def init_logging(level_name, name, root=False) -> logging.Logger:
    """
    Initialize the logging object
    """

    try:
        logger = logging.getLogger(name)

        level = logging.getLevelName(level_name)

        logger.setLevel(level)

        if not logger.hasHandlers():
            handler = logging.FileHandler(conf['log_file'])
            handler.setLevel(level)
            stdout_handler = logging.StreamHandler(sys.stdout)
            stdout_handler.setLevel(level)

            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            stdout_handler.setFormatter(formatter)

            logger.addHandler(handler)
            logger.addHandler(stdout_handler)

        return logger

    except Exception as e:
        print(e)
        sys.exit()
