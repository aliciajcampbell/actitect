import ast
import contextlib
import csv
import datetime
import json
import logging
import logging.config
import os
import re
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Dict
from typing import Union

import joblib
import numpy as np
import pandas as pd
import scipy.stats as stats
import yaml
from colorama import Fore, Style
from colorama import init as colorama_init
from openmovement.load import CwaData
from skopt.space import Dimension
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

try:
    from numba import njit as _numba_njit
    NUMBA_AVAILABLE = True
except ImportError:        # user did not install numba ⇒ graceful fallback
    NUMBA_AVAILABLE = False

    # no-op stand-in that accepts the same signature as njit(...)
    def _numba_njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

__all__ = [
    'get_experiment_root',
    'detect_csv_delimiter',
    'mute_matplotlib_logger',
    'setup_logging',
    'read_meta_csv_to_df',
    'load_yaml_file',
    'map_log_level',
    'combine_diary_times',
    'load_cwa_to_df',
    'standardize_sleep_diary',
    'extract_CGN_ID_from_path',
    'check_make_dir',
    'warning_handler',
    'dump_to_json',
    'read_from_json',
    'Timer',
    'df_astype_inplace',
    'CustomJsonEncoder',
    'concat_dicts_with_times',
    'fmt_dict',
    'dict_minus_key',
    'independent_stat_significance_test',
    'tqdm_joblib',
    'custom_tqdm',
    'get_file_extension',
    'get_num_workers',
    'str_to_snake_case',
    'split_df_in_chunks',
    'compare_dicts',
    'subdict',
    'nested_dict_iterator',
    'trim_whitespace_df',
    'list_subdirectories',
    'compute_pearson_ci',
    'compute_mean_std_ci',
    'optional_njit'
]

logger = logging.getLogger(__name__)
colorama_init(autoreset=True)


def get_experiment_root():
    _json = Path(__file__).parents[1].joinpath('config/experiment_root.json')
    assert _json.is_file(), f"'{_json}' is not a file."
    return Path(read_from_json(_json).get('EXPERIMENT_ROOT', 'none'))


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
            _log_config['handlers']['file'].update({'filename': _unique_log_path})
        else:  # disable file logging and only print console logging
            _log_config['loggers']['aktiRBD'].update({'handlers': ['console']})
            _log_config['root'].update({'handlers': ['console']})
            _ = _log_config['handlers'].pop('file')
        logging.config.dictConfig(_log_config)

    sys.excepthook = _log_exceptions


@contextmanager
def custom_tqdm(total, position=0, leave=True, disable=False, color=Fore.MAGENTA, style=Style.RESET_ALL):
    bar_format = f"{color}{{l_bar}}{{bar}}{{r_bar}}{style}"
    # bar_format = '{desc} [ {bar:100} ] {n_fmt}/{total_fmt} {percentage:3.0f}%'
    _ascii = "□■"

    with logging_redirect_tqdm(), tqdm(total=total, position=position, leave=leave, disable=disable, ascii=_ascii,
                                       bar_format=bar_format) as pbar:
        yield pbar


# i/o related:

def detect_csv_delimiter(file_path):
    with open(file_path, 'r', newline='', encoding='utf-8') as file:
        sample = file.read(2048)
        sniffer = csv.Sniffer()
        try:
            dialect = sniffer.sniff(sample)
            if dialect.delimiter not in [';', ',', '\t', '|']:  # restrict to common delimiters
                raise ValueError(f"unexpected delimiter: {dialect.delimiter}")
            return dialect.delimiter
        except Exception as e:
            logger.warning(f"failed to detect delimiter: {e}. Defaulting to ';'")
            return ';'


def read_meta_csv_to_df(path_to_csv: Path, exclude: bool = False, verbose: bool = True):
    delimiter = detect_csv_delimiter(path_to_csv)
    meta_df = pd.read_csv(path_to_csv, delimiter=delimiter)
    meta_df.columns = meta_df.columns.str.strip()
    if '#' in meta_df.columns:
        meta_df.set_index('#', inplace=True)

    # format timestamps if needed
    if 'timestamps' in meta_df.columns:
        meta_df['timestamps'] = meta_df['timestamps'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    filtered_cols = [_col for _col in meta_df.columns if 'Unnamed' not in _col]
    meta_df = meta_df[filtered_cols]

    # strip whitespaces in strings if present
    meta_df = meta_df.map(lambda x: x.strip() if isinstance(x, str) else x)

    # optionally exclude rows
    if exclude and 'exclude' in meta_df.columns:
        excluded_rows = meta_df[meta_df['exclude'] == 1]
        dropped_ids = excluded_rows['ID'].tolist()
        if verbose:
            logger.info(f"Dropping {len(excluded_rows)} row(s) with exclude == 1. IDs: {dropped_ids}")
        meta_df = meta_df[meta_df['exclude'] != 1]

    # ensure required columns exist:
    required_cols = {'filename', 'ID'}
    missing_cols = required_cols - set(meta_df.columns)
    if missing_cols:
        raise ValueError(f"Metadata file is missing required columns: {missing_cols}")
    if 'record_ID' not in meta_df.columns:
        meta_df['record_ID'] = None

    # drop last row if it is a summary row (simply check if last row contains a valid filename or not)
    def __row_has_valid_filename(row):
        from aktiRBD.actimeter import SUPPORTED_FILETYPES
        return any(str(row['filename']).endswith(ext) for ext in SUPPORTED_FILETYPES)
    if not __row_has_valid_filename(meta_df.iloc[-1]):
        meta_df = meta_df.iloc[:-1].copy()

    def _assign_record_ID(group: pd.DataFrame, ID_value: str):
        """ Assigns unique recording IDs for patients with multiple recordings.
        Preserves user-defined values where present. """
        group.insert(group.columns.get_loc('filename') + 1, 'ID', ID_value)
        if group.shape[0] == 1:
            return group  # only one recording for ID, no change needed
        if group['record_ID'].isna().all():  # multiple recordings but no user-define record_IDs
            group['record_ID'] = [f"rec{i + 1}" for i in range(group.shape[0])]  # auto-complete record_IDs
        elif group['record_ID'].isna().any():  # some exist, but not all
            logger.warning(f"`record_ID` values are only partially defined for ID={ID_value}."
                           f" Using auto-completion for missing values. ")
            existing_ids = set(group['record_ID'].dropna())  # keep defined IDs
            new_ids = [f"rec{i + 1}" for i in range(1, group.shape[0] + 1) if f"rec{i + 1}" not in existing_ids]
            group.loc[group['record_ID'].isna(), 'record_ID'] = new_ids[:group['record_ID'].isna().sum()]

        return group

    meta_df = meta_df.groupby('ID', group_keys=False).apply(
        lambda g: _assign_record_ID(g, g.name), include_groups=False)

    assert not meta_df.duplicated(subset=['ID', 'record_ID']).any(), \
        "Duplicate combinations of 'ID' and 'record_ID' found in meta file."
    return meta_df


def load_yaml_file(path):
    with open(path, "r") as stream:
        try:
            _cfg = yaml.safe_load(stream)
            return _cfg
        except yaml.YAMLError as exc:
            logger.error(f" (io): error ({exc}) encountered while loading config {path}")


def load_cwa_to_df(file_path):
    """
    loads the actiwatch accelerometer as df.
    :param file_path:
    :return: pandas.df with columns ['time', 'accel_x', 'accel_y', 'accel_z', 'light', 'temperature']
    """
    with CwaData(
            file_path,
            include_gyro=True, include_temperature=True,
            include_light=True, include_mag=True,
            include_time=True, include_accel=True,
    ) as cwa_data:
        df = cwa_data.get_samples()
        return df


def map_log_level(log_level_key: str):
    mapping_dict = {'debug': logging.DEBUG, 'meta': logging.INFO, 'warning': logging.WARNING,
                    'error': logging.ERROR, 'critical': logging.CRITICAL}
    return mapping_dict[log_level_key]


# processing and formatting:
def combine_date_time(date, time):
    # TODO: -> directly include together with combine_diary_times in standardize dataframe method!
    # print('date/time', date, time)
    # print('\n combine:', datetime.datetime.combine(date, time))
    try:
        return datetime.datetime.combine(date, time)
    except:
        return np.nan


def combine_diary_times(row):
    """
    adds a date to the sleep log times.
    :param row:
    :return:
    """
    # date = row[0]

    date = row['date_start']
    # print('date:', date)
    # print('date:', date, row[0])
    for i, k in enumerate(row[1:]):
        # print(i, k)
        try:
            days = i // 2
            if i % 2 == 1 and 0 <= k.hour <= 5:
                days += 1
            row.iloc[i + 1] = combine_date_time(date + datetime.timedelta(days=days), k)
        except:
            row.iloc[i + 1] = np.NaN

    return row


def standardize_sleep_diary(df_in):
    head = ['ID', 'Visite', 'date_start']
    times = ['time_wakeup_1', 'time_gosleep_1', 'time_wakeup_2', 'time_gosleep_2', 'time_wakeup_3', 'time_gosleep_3',
             'time_wakeup_4', 'time_gosleep_4', 'time_wakeup_5', 'time_gosleep_5', 'time_wakeup_6', 'time_gosleep_6',
             'time_wakeup_7', 'time_gosleep_7', 'time_wakeup_8', 'time_gosleep_8']
    try:

        missing_columns = [col for col in head + times if col not in df_in.columns]
        if missing_columns:
            raise ValueError(f"Missing expected columns: {', '.join(missing_columns)}")

        df_in = df_in[head + times]
        df_in = df_in.fillna('NaT')

        for col in times:
            if df_in[col].dtype != 'datetime64[ns]':
                df_in.loc[:, col] = pd.to_datetime(df_in[col], format='%H:%M:%S').dt.time

        df_out = pd.DataFrame(columns=head + times)
        df_out[head] = df_in[head]
        times_indices = [df_in.columns.get_loc(col) for col in times]
        df_out.iloc[:, times_indices] = df_in.iloc[:, times_indices].map(extract_time)

        df_out.iloc[:, 2:] = df_out.iloc[:, 2:].apply(combine_diary_times, axis=1)

        # drop all rows where ID=NaN:
        df_out = df_out.dropna(subset=['ID'])

        return df_out

    except Exception as e:
        raise UserWarning(f"Error while standardizing sleep diary DataFrame: {str(e)}")


def extract_time(timestamp):
    if isinstance(timestamp, pd.Timestamp):
        return timestamp.time()
    elif isinstance(timestamp, datetime.datetime):
        if not pd.isna(timestamp):
            return timestamp.time()
        else:
            return None
    elif isinstance(timestamp, datetime.time):
        return timestamp
    elif pd.isna(timestamp):
        return None
    else:
        logger.warning(f" unexpected dtype encountered in 'convert_to_datetime': {timestamp} of {timestamp.dtype}")
        try:
            return pd.to_datetime(timestamp).time()
        except:
            raise ValueError(f" Unknown value in 'convert_to_time', check excel formatting.")


def extract_CGN_ID_from_path(file_path: Union[str, Path], reg_ex=('RBD', 'HC-RBD')):
    if isinstance(file_path, Path):
        file_path = str(file_path)
    elif isinstance(file_path, str):
        pass
    else:
        raise TypeError(f"file path must be of type 'str' or 'PosixPath', got {type(file_path)}")

    _file_wo_path = file_path.split('/')[-1]
    match = re.search('|'.join(reg_ex), _file_wo_path)
    if match:
        _num = _file_wo_path[match.end():match.end() + 4]
        _cls = match.group(0)
        return _cls + _num
    else:
        raise UserWarning(f" no reg_ex found in {file_path}.")


def check_make_dir(dir_path: Path, use_existing=False, verbose=True, time_extension: bool = True):
    """
    Ensure the directory at `dir_path` exists. If not, create it. If it exists and `use_existing` is False,
    a new directory with an incremented suffix ('_1', '_2', etc.) is created.

    Parameters:
        :param: dir_path: (Path) path object of the directory to check/create.
        :param: use_existing: (bool, Optional) If True, use the existing directory if it exists.
                    If False, create a new directory with a numeric suffix if it exists. Default is False.
        :param: verbose: (bool, Optional) If True, logs messages about the directory creation or reuse. Default is True.
        :param: time_extension (bool, Optional) If True, will add current time as extension instead of 1,2,3...

    Returns:
        :return: (Path) the path objected pointing to the created or reused directory.

    Notes:
    - If the directory already exists and `use_existing` is True, the existing directory is reused.
    - If the directory already exists and `use_existing` is False, a new directory with a suffix is created.
    """
    if not dir_path.exists():
        try:
            # os.makedirs(dir_path)
            dir_path.mkdir(parents=True, exist_ok=False)
            if verbose:
                logger.debug(f"(io): dir. '{dir_path}' created.")
            return dir_path
        except OSError as e:
            logger.error(f"(io): error creating dir. '{dir_path}': {e}")
    else:
        if use_existing:
            if verbose:
                logger.info(f"(io): dir '{dir_path}' exists and will be used."
                            f" Set 'use_existing' to False, to create new dir to avoid potential overwriting.")
            return dir_path

        else:

            if time_extension:
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss")
                new_dir_name = f"{dir_path.name}_{timestamp}"
                new_dir_path = dir_path.parent / new_dir_name
                try:
                    new_dir_path.mkdir(parents=True, exist_ok=False)
                    if verbose:
                        logger.warning(f"(io): Directory '{dir_path}' already exists. Created '{new_dir_path}'.")
                    return new_dir_path
                except OSError as e:
                    logger.error(f"(io): Error creating directory '{new_dir_path}': {e}")
                    raise

            # Determine the numeric suffix to append to the directory name
            suffix = 1
            # new_dir_path = f"{dir_path}_{suffix}"
            new_dir_path = dir_path.parent.joinpath(f"{dir_path.name}_{suffix}")

            # while os.path.exists(new_dir_path):
            while new_dir_path.exists():
                suffix += 1
                # new_dir_path = f"{dir_path}_{suffix}"
                new_dir_path = dir_path.parent.joinpath(f"{dir_path.name}_{suffix}")
            try:
                # os.makedirs(new_dir_path)
                new_dir_path.mkdir(parents=True, exist_ok=False)
                logger.warning(f"(io): dir. '{dir_path}' already exists. created {new_dir_path}")
                return new_dir_path
            except OSError as e:
                logger.error(f"(io): error creating dir. '{new_dir_path}': {e}")
                raise


def fmt_dict(input_dict, digits=2):
    """
    Returns a new dictionary with the values formatted to a specified number of decimal places
    if they are numbers, or left unchanged if they are strings.

    Parameters:
    input_dict (dict): The dictionary whose values need to be formatted.
    digits (int): The number of decimal places to format the values to.

    Returns:
    dict: A new dictionary with formatted values.
    """
    formatted_dict = {}
    for k, v in input_dict.items():
        if isinstance(v, (int, float)):
            formatted_dict[k] = round(v, digits)
        elif isinstance(v, str):
            formatted_dict[k] = v
        else:
            formatted_dict[k] = v
    return formatted_dict


def dict_minus_key(in_dict, keys_to_remove):
    """
    Returns a copy of in_dict without the key(s) specified in keys_to_remove.
    :param in_dict:
    :param keys_to_remove:
    :return:
    """
    return {key: in_dict[key] for key in in_dict if key not in keys_to_remove}


def warning_handler(message, category, filename, lineno, file=None, line=None):
    warning_msg = f" Warning in {filename}, line {lineno}: {category.__name__}: {message}"
    logger.warning(warning_msg)


@contextmanager
def Timer():
    start_time = time.perf_counter()
    yield lambda: np.round(time.perf_counter() - start_time, 2)


def df_astype_inplace(df: pd.DataFrame, dct: Dict):
    df[list(dct.keys())] = df.astype(dct)[list(dct.keys())]


class CustomJsonEncoder(json.JSONEncoder):
    """Serializes numpy and pandas objects as JSON."""

    def default(self, obj):
        if isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, pd.Timestamp):
            if pd.isna(obj):  # Handle NaT (missing datetime)
                return None
            return obj.strftime('%Y-%m-%d %H:%M:%S')
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.floating):
            if np.isnan(obj):
                return None  # Serialize NaN as JSON null
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, Dimension):
            return self._serialize_dimension(obj)
        elif callable(obj):
            return str(obj)
        else:
            return super().default(obj)

    @staticmethod
    def _serialize_dimension(obj):
        """Serialize skopt Dimension objects to a dictionary."""
        serialized = {'name': obj.name, 'kwargs': {'type': obj.__class__.__name__}}
        for attribute in dir(obj):
            # filter out private and built-in attributes
            if not attribute.startswith('_') and not callable(getattr(obj, attribute)):
                serialized['kwargs'][attribute] = str(getattr(obj, attribute))
        return serialized


def concat_dicts_with_times(dict_a, dict_b):
    result = {}
    for key in set(dict_a.keys()).union(dict_b.keys()):
        value_a = dict_a.get(key)
        value_b = dict_b.get(key)

        if value_a is None and value_b is None:
            result[key] = None
        elif value_a is None:
            result[key] = value_b
        elif value_b is None:
            result[key] = value_a
        else:
            result[key] = value_a + value_b if 'time' in key else value_b
    return result


def dump_to_json(data_dict, file_path, indent=4, json_encoder=CustomJsonEncoder):
    """
    Save a dictionary to a JSON file.

    Parameters::
        :param data_dict: (dict) Dictionary to be saved.
        :param file_path: (str or Path) Path to the file where the JSON will be saved.
        :param indent: (int, optional) Indentation level for the JSON file. Default is 4.
        :param json_encoder: (json.JSONEncoder, optional) Custom JSON encoder, if needed.
    """
    file_path = Path(file_path)
    with open(file_path, 'w') as f:
        json.dump(data_dict, f, indent=indent, cls=json_encoder)


def read_from_json(file_path):
    """
    Read a saved JSON file as dict.

    Parameters::
        :param file_path: (str or Path) Path to the file where the JSON will be saved.
    """
    file_path = Path(file_path)
    with open(file_path, 'r') as f:
        _dict = json.load(f)
    return _dict


def independent_stat_significance_test(distribution_a: np.ndarray, distribution_b: np.ndarray, names=('a', 'b')):
    """Quantify the difference between two distributions. Assumes independence of both variables

    Parameters:
        :param distribution_a: (np.ndarray) of shape (n_samples_a,) representing the distribution A.
        :param distribution_b: (np.ndarray) of shape (n_samples_b,) representing the distribution B.
        :param names: (tuple of strings) names for the two distributions to store results. Optional, default: ('a', 'b')

    Returns:
        :return: (dict)
            Dictionary containing several metrics.

    Metrics:
        - normality: Tests whether both distributions follow a normal distribution.
            - shapiro stat/p: Shapiro-Wilk test stat [0,1] (normal to not) and p_value [0,1]
             (smaller=stronger rejection of normal hypothesis), might be inaccurate for large sample size.
            - kolmogorov stat/p: Kolmogorov-Smirnov test, same metrics as above but for stat 0 indicates normal
            - anderson stat/thres(5% cl): Anderson-Darling test, stat [0,inf) (smaller=normal) and 5% CL threshold
        - difference:
            - 'T-test': Independent t-test, suited to compare normal distributions.
                - stat/z: the difference in units of std,
                - p: p_value to reject null-hypothesis of no significance, i.e. smaller=more significant.
                    Typically:
                        p ≤   0.05:    *   -> less than 5% CL that distributions are not sign. different
                        p ≤   0.01:   **   ->    -"-    1% CL              -"-
                        p ≤  0.001:  ***   ->    -"-  0.1% CL              -"-
                        p ≤ 0.0001: ****   ->    -"- 0.01% CL              -"-
                - r: effect size: small (0.1) to large (0.5) effect of significance
    """
    # test if distributions are normal or not:
    results = {'normality': {}, 'difference': {}}
    for data, name in zip((distribution_a, distribution_b), names):
        shapiro_stat, shapiro_p_value = stats.shapiro(data)  # stat: 0 (non-normal) 1 (normal)
        k_stat, k_p_value = stats.kstest(data, 'norm',
                                         args=(data.mean(), data.std(ddof=1)))  # stat: 0 (normal) 1 (non-normal)
        ad_test = stats.anderson(data)
        ad_stat, critical_value_at_5_cl = ad_test.statistic, ad_test.critical_values[2]

        results['normality'].update({f"{name}": {
            'shapiro': {'stat': shapiro_stat, 'p': shapiro_p_value},
            'kolmogorov': {'stat': k_stat, 'p': k_p_value},
            'anderson': {'stat': ad_stat, 'thres_5': critical_value_at_5_cl},
        }})

    n1, n2 = len(distribution_a), len(distribution_b)

    # t-test for normal distributions:
    t_stat, p_val_t = stats.ttest_ind(distribution_a, distribution_b, equal_var=False)
    df = (n1 - 1) + (n2 - 1)
    r_t = (t_stat ** 2 / (t_stat ** 2 + df)) ** 0.5  # effect size: small (.1) - large (.5) difference
    results['difference'].update({
        'T-test': {'stat/z': t_stat, 'p': p_val_t, 'r': r_t}
    })

    # Mann-Whitney U (or Wilcoxon rank-sum) for arbitrary distribution:
    u_stat, p_val_u = stats.mannwhitneyu(distribution_a, distribution_b, alternative='two-sided')
    mean_u = n1 * n2 / 2
    std_u = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    z = (u_stat - mean_u) / std_u  # z-score: significance in std units
    r_u = z / np.sqrt(n1 + n2)  # effect size: small (.1) - large (.5) difference

    results['difference'].update({
        'Mann-Whitney-U': {'stat': u_stat, 'p': p_val_u, 'z': z, 'r': r_u}
    })

    return results


@contextlib.contextmanager
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


def get_file_extension(file_path, *, include_dot=True, all_extensions=False):
    """ For a given file path, return the file extension(s).

    Parameters:
        :param file_path: (Path or str): The path of the file.
        :param include_dot: (bool, Optional) If True, include the leading dot in the return str. Default is True.
        :param all_extensions: (bool, Optional) If True, returns all extensions as list.
            Default is False: return only last extension as str.
    Returns:
        :return: str or list[str] containing the last or all file extensions.
    """
    # Ensure file_path is a Path object
    if not isinstance(file_path, Path):
        file_path = Path(file_path)

    if all_extensions:  # collect all suffixes
        suffixes = file_path.suffixes
        if not include_dot:
            suffixes = [suffix.lstrip('.') for suffix in suffixes]
        return suffixes
    else:  # only last suffix
        suffix = file_path.suffix
        if not include_dot:
            suffix = suffix.lstrip('.')
        return suffix


def get_num_workers(n_jobs: int):
    """ For a given platform, calculate the number of cpu workers given the 'n_jobs' argument."""
    n_available = os.cpu_count()
    if n_jobs == -1 or n_jobs >= n_available:
        return n_available
    else:
        return n_jobs


def str_to_snake_case(text: str):
    """" transform a given str to snake_case format """
    return re.sub(r'\s+', '_', text.strip().lower())


def split_df_in_chunks(df: pd.DataFrame, days_per_chunk: int):
    """ Split a DataFrame into chunks containing a specified number of days per chunk.
    Parameters:
        :param df: (DataFrame) the input df, must have a DatetimeIndex.
        :param days_per_chunk: (int) The number of days each chunk should span.
    Returns:
        :return: (List[DataFrame]) List of df's, each containing data for up to 'days_per_chunk' days.
    """

    assert isinstance(df.index, pd.DatetimeIndex), "Index must be a DatetimeIndex."
    num_days = (df.index.max() - df.index.min()).days + 1  # number of unique days spanned by df

    if num_days <= days_per_chunk:
        return [df]
    else:
        chunks = []
        start_date = df.index.min()
        while start_date < df.index.max():
            end_date = start_date + pd.Timedelta(days=days_per_chunk - 1)
            chunk = df.loc[start_date:end_date]
            chunks.append(chunk)
            start_date = end_date + pd.Timedelta(days=1)
        return chunks


def compare_dicts(d1, d2, path=""):
    """Recursively compare two dictionaries and return differences."""
    differences = {}
    for key in d1.keys() | d2.keys():  # Union of keys
        if key not in d1:
            differences[path + key] = ("<MISSING>", d2[key])
        elif key not in d2:
            differences[path + key] = (d1[key], "<MISSING>")
        elif isinstance(d1[key], dict) and isinstance(d2[key], dict):
            sub_diffs = compare_dicts(d1[key], d2[key], path + key + ".")
            differences.update(sub_diffs)
        elif d1[key] != d2[key]:
            differences[path + key] = (d1[key], d2[key])
    return differences


def subdict(original: dict, keys_to_keep: list):
    """ Returns a new sub-dictionary containing only the specified keys from the original dictionary.

    Parameters:
        original (dict): The original dictionary to filter.
        keys_to_keep (list): A list of keys to retain in the new dictionary.

    Returns:
        dict: A new dictionary with only the specified keys.
    """
    missing_keys = [k for k in keys_to_keep if k not in original]
    if missing_keys:
        raise KeyError(f"The following keys are missing in the original dictionary: {missing_keys}")

    return {k: original[k] for k in keys_to_keep}


def nested_dict_iterator(nested_dict: dict, iteration_depth: int, current_depth=1, keys=None):
    """ A utility function to iterate over nested dictionaries up to a specified depth.
    Parameters:
        :param nested_dict: (dict) The nested dictionary to iterate over.
        :param iteration_depth: (int) The depth of keys to return in the iterator.
        :param current_depth: (int) The current depth during recursion.
        :param keys: (list) The accumulated keys so far.
    Yields:
        tuple: A tuple where the first element is a list of keys at the specified depth
               and the second element is the value at the next level. """
    keys = keys or []
    if current_depth < iteration_depth:
        if isinstance(nested_dict, dict):
            for k, v in nested_dict.items():
                yield from nested_dict_iterator(v, iteration_depth, current_depth + 1, keys + [k])
    else:
        yield keys, nested_dict


def trim_whitespace_df(df: pd.DataFrame):
    """Removes leading and trailing whitespaces from column names and string values in a DataFrame."""
    df.columns = df.columns.str.strip()  # trim column names
    return df.map(lambda x: x.strip() if isinstance(x, str) else x)  # trim string values


def list_subdirectories(base_path: Path, depth: int, stop_names: set = None):
    """  List subdirectories of base_path at a specified depth or earlier if a directory's name
    is in stop_names. When a directory is encountered whose name is in stop_names, that directory
    is added to the result and its children are not searched.
    Parameters:
      base_path (Path): The starting directory.
      depth (int): The maximum depth to search (depth=1 returns immediate children).
      stop_names (set, optional): A set of directory names at which to stop recursion.
                                  For example: {"early_stopping", "no_early_stopping"}
    Returns:
      list of Path: Subdirectories found, expressed as paths relative to base_path. """
    stop_names = stop_names or set()
    result = []

    def _traverse(path: Path, current_depth: int):
        # If we've reached the maximum depth or the directory name is in stop_names,
        # add it to the result (if it's a directory) and do not descend further.
        if current_depth == depth or path.name in stop_names:
            if path.is_dir():
                result.append(path)
            return
        # Otherwise, traverse its children.
        for child in path.iterdir():
            if child.is_dir():
                _traverse(child, current_depth + 1)

    _traverse(base_path, 0)
    # Convert the absolute paths in result to paths relative to base_path.
    return [p.relative_to(base_path) for p in result]


def compute_pearson_ci(r: float, n: int, confidence_level: float = .95):
    """Compute confidence interval for Pearson correlation using Fisher z-transform."""
    assert n >= 10, f" must have at least 10 samples to estimate Fisher z-transform, got {n}."

    z = np.arctanh(r)  # Fisher z-transform
    se = 1 / np.sqrt(n - 3)  # standard error
    std_r = (1 - r ** 2) / np.sqrt(n - 3)  # estimation using delta method
    z_crit = stats.norm.ppf((1 + confidence_level) / 2)
    z_ci_lower = z - z_crit * se
    z_ci_upper = z + z_crit * se
    return np.tanh(z_ci_lower), np.tanh(z_ci_upper), std_r


def compute_mean_std_ci(data: np.ndarray, confidence_level: float):
    """ Compute the mean, standard deviation, and confidence interval for the mean of a dataset.
    This function calculates:
      - The mean of the input data
      - The sample standard deviation (using N−1 in the denominator)
      - A confidence interval (CI) around the mean, assuming a t-distribution
    Parameters
        :param data: (np.ndarray) The input data array for which the statistics will be computed.
        :param confidence_level: (float) The desired confidence level (e.g., 0.95 for a 95% confidence interval).

    Returns
        :return mean_val: (float) The sample mean of the data.
        :return std_val: (float) The sample standard deviation of the data (ddof=1).
        :return ci_lower: (float or nan)  The lower bound of the confidence interval around the mean.
            Returns nan if n <= 1.
        :return ci_upper: (float or nan) The upper bound of the confidence interval around the mean.
            Returns nan if n <= 1.
        :return margin: (float or nan)  The margin of error used to compute the confidence interval.
            Returns nan if n <= 1. """
    n = len(data)
    mean_val, std_val = np.mean(data), np.std(data, ddof=1)
    margin, ci_lower, ci_upper = np.nan, np.nan, np.nan
    if n > 1:
        sem = stats.sem(data)  # Standard Error of the Mean
        margin = sem * stats.t.ppf((1 + confidence_level) / 2., n - 1)
        ci_lower = mean_val - margin
        ci_upper = mean_val + margin
    return mean_val, std_val, ci_lower, ci_upper, margin

def optional_njit(*args, **kwargs):
    """Decorate a function with numba.njit if Numba is present,
    otherwise leave it as plain Python."""
    return _numba_njit(*args, **kwargs)
