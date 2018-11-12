import logging
import sys
import os

log_fn = "./facerec.log"
logging.basicConfig(level=logging.INFO,
                    format='%s(asctime)s %(message)s',
                    datefmt='%d/%b/%Y %H:%M:%S',
                    filename=log_fn)

log_obj = logging


def log_init():
    if os.path.isfile(log_fn):
        os.remove(log_fn)


def log_print(log_str):
    sys.stdout.write(log_str)
    if log_str[0] == '\r':
        log_obj.info(log_str[1:])
    else:
        log_obj.info(log_str[:])
