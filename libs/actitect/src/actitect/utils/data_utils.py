import datetime
import logging.config
import os
from typing import Dict

import numpy as np
import pandas as pd

try:
    from numba import njit as _numba_njit

    NUMBA_AVAILABLE = True
except ImportError:  # numba not installed
    NUMBA_AVAILABLE = False


    def _numba_njit(*args, **kwargs):  # dummy decorator that accepts the same signature as njit(...)
        def decorator(func):
            return func

        return decorator

__all__ = [
    'standardize_sleep_diary',
    'df_astype_inplace',
    'get_num_workers',
    'split_df_in_chunks',
    'trim_whitespace_df',
    'optional_njit',
]

logger = logging.getLogger(__name__)


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
        logger.warning(f"unexpected dtype encountered in 'convert_to_datetime': {timestamp} of {timestamp.dtype}")
        try:
            return pd.to_datetime(timestamp).time()
        except:
            raise ValueError(f"Unknown value in 'convert_to_time', check excel formatting.")


def df_astype_inplace(df: pd.DataFrame, dct: Dict):
    df[list(dct.keys())] = df.astype(dct)[list(dct.keys())]


def get_num_workers(n_jobs: int):
    """ For a given platform, calculate the number of cpu workers given the 'n_jobs' argument."""
    n_available = os.cpu_count()
    if n_jobs == -1 or n_jobs >= n_available:
        return n_available
    else:
        return n_jobs


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


def trim_whitespace_df(df: pd.DataFrame):
    """Removes leading and trailing whitespaces from column names and string values in a DataFrame."""
    df.columns = df.columns.str.strip()  # trim column names
    return df.map(lambda x: x.strip() if isinstance(x, str) else x)  # trim string values


def optional_njit(*args, **kwargs):
    """Decorate a function with numba.njit if Numba is present,
    otherwise leave it as plain Python."""
    return _numba_njit(*args, **kwargs)


def standardize_sleep_diary(df_in: pd.DataFrame) -> pd.DataFrame:
    def __combine_diary_times(row):

        def _combine_date_time(_date, _time):
            try:
                return datetime.datetime.combine(_date, _time)
            except:
                return np.nan

        date = row['date_start']
        # print('date:', date)
        # print('date:', date, row[0])
        for i, k in enumerate(row[1:]):
            # print(i, k)
            try:
                days = i // 2
                if i % 2 == 1 and 0 <= k.hour <= 5:
                    days += 1
                row.iloc[i + 1] = _combine_date_time(date + datetime.timedelta(days=days), k)
            except:
                row.iloc[i + 1] = np.NaN

        return row

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

        df_out.iloc[:, 2:] = df_out.iloc[:, 2:].apply(__combine_diary_times, axis=1)

        # drop all rows where ID=NaN:
        df_out = df_out.dropna(subset=['ID'])

        return df_out

    except Exception as e:
        raise UserWarning(f"Error while standardizing sleep diary DataFrame: {str(e)}")
