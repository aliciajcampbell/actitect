import datetime
import logging
import logging.config
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

import joblib
import numpy as np
import yaml
from colorama import Fore, Style
from colorama import init as colorama_init
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from .io_utils import check_make_dir

logger = logging.getLogger(__name__)
colorama_init(autoreset=True)

__all__ = [
    'map_log_level', 'setup_logging', 'mute_matplotlib_logger',
    'custom_tqdm', 'tqdm_joblib', 'warning_handler', 'Timer']


@contextmanager
def mute_matplotlib_logger():
    matplotlib_logger = logging.getLogger('matplotlib')
    original_level = matplotlib_logger.getEffectiveLevel()

    try:
        matplotlib_logger.setLevel(logging.CRITICAL)
        yield
    finally:
        matplotlib_logger.setLevel(original_level)


def setup_logging(config_path=Path(__file__).parents[1].joinpath('config/logging_config.yaml'),
                  log_file_path: Path = None, level: str = None):
    def _replace_level_keys_with_uppercase(d, new_value):
        """Recursively replace all 'level' keys with a new value converted to uppercase in a nested dictionary."""
        if not isinstance(d, dict):
            return d
        uppercase_value = new_value.upper()  # convert the new value to uppercase
        for key, value in list(d.items()):  # use list to avoid RuntimeError from modifying dict during iteration
            if key == 'level':
                d[key] = uppercase_value
            elif isinstance(value, dict):
                _replace_level_keys_with_uppercase(value, new_value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        _replace_level_keys_with_uppercase(item, new_value)
        return d

    def _log_exceptions(ex_cls, ex, tb):
        text = ''.join(traceback.format_exception(ex_cls, ex, tb))
        logger.critical(f"Exception: {text}")

    with open(config_path, 'r') as f:
        _log_config = yaml.safe_load(f.read())
        if level:
            _log_config = _replace_level_keys_with_uppercase(_log_config, level)

        if log_file_path:  # set output log file location
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%Hh%Mm%Ss')
            _unique_log_path = log_file_path.with_name(f"{log_file_path.stem}_{timestamp}{log_file_path.suffix}")
            check_make_dir(_unique_log_path.parent, use_existing=True)
            _log_config['handlers']['file'].update({'filename': _unique_log_path})
        else:  # disable file logging and only print console logging
            _log_config['loggers']['actitect'].update({'handlers': ['console']})
            _log_config['root'].update({'handlers': ['console']})
            _ = _log_config['handlers'].pop('file')
        logging.config.dictConfig(_log_config)

    sys.excepthook = _log_exceptions


@contextmanager
def custom_tqdm(total, position=0, leave=True, disable=False, color=Fore.MAGENTA, style=Style.RESET_ALL):
    bar_format = f"{color}{{l_bar}}{{bar}}{{r_bar}}{style}"
    _ascii = "□■"
    with logging_redirect_tqdm(), tqdm(total=total, position=position, leave=leave, disable=disable, ascii=_ascii,
                                       bar_format=bar_format) as pbar:
        yield pbar


def map_log_level(log_level_key: str):
    mapping_dict = {'debug': logging.DEBUG, 'meta': logging.INFO, 'warning': logging.WARNING,
                    'error': logging.ERROR, 'critical': logging.CRITICAL}
    return mapping_dict[log_level_key]


@contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def warning_handler(message, category, filename, lineno, file=None, line=None):
    warning_msg = f"Warning in {filename}, line {lineno}: {category.__name__}: {message}"
    logger.warning(warning_msg)

@contextmanager
def Timer():
    start_time = time.perf_counter()
    yield lambda: np.round(time.perf_counter() - start_time, 2)
