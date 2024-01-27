# Copyright © 2023-2024 Apple Inc.
import logging
import sys
import time


def init_logging(level: int):

    logging_format = ('%(asctime)s %(levelname)s: '
                      '%(filename)s:%(lineno)d: %(message)s')
    # Use UTC.
    logging.Formatter.converter = time.gmtime
    logging.basicConfig(stream=sys.stdout, level=level, format=logging_format)
