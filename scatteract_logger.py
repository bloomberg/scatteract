# -*- coding: utf-8 -*-

try:
    from functools import lru_cache
except ImportError:
    from backports.functools_lru_cache import lru_cache

import logging
import logging.handlers

@lru_cache(maxsize=1)
def get_logger():
    LOGGING_MSG_FORMAT  = '[%(levelname)s] [%(asctime)s] [%(filename)s:%(lineno)s] : %(message)s'
    LOGGING_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

    full_file_name = "scatteract.log"
    bblogger = logging.getLogger(__name__)
    bblogger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(LOGGING_MSG_FORMAT, datefmt=LOGGING_DATE_FORMAT)

    # Add the log message handler to the logger
    handler = logging.handlers.TimedRotatingFileHandler(full_file_name,
                                                        when='midnight', interval=1, backupCount=5)
    handler.suffix = '%Y%m%d.log'
    handler.setFormatter(fmt)
    bblogger.addHandler(handler)

    return bblogger
