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
    'standardize_sleep_diary', 'df_astype_inplace', 'get_num_workers', 'split_df_in_chunks',
    'trim_whitespace_df', 'optional_njit', 'assert_valid_df', 'extract_segments', 'handle_duplicate_timestamps',
    'infer_mean_sample_rate', 'mmap_like', 'mmap2df', 'copy2mmap'
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

def assert_valid_df(df: pd.DataFrame, nan_threshold: float = 0.0):
    """ Validate the structure and data quality of a DataFrame.
    Parameters:
        :param df: (pd.DataFrame) with a DatetimeIndex containing required columns 'x', 'y', 'z', and 'time'.
        :param nan_threshold: (float) Maximum allowed fraction of rows with NaN values. Defaults to 0.0.
    Raises:
        AssertionError: If 'df' is not a pd.DataFrame, lacks a DatetimeIndex, or is missing required columns.
        UserWarning: If the fraction of NaN rows exceeds 'nan_threshold'."""
    assert isinstance(df, pd.DataFrame), f"Passed argument is not a pd.DataFrame instance, got: {df}"
    assert isinstance(df.index, pd.DatetimeIndex), "DataFrame index is not a datetime index."

    _required_columns = {'x', 'y', 'z', 'time'}
    _accepted_extra_columns = {'wear', 'sptw', 'sleep_bout', 'light', 'temperature', 'button', 'battery'}
    _df_columns = set(df.columns).union({'time'})
    missing_required_columns = _required_columns - _df_columns
    assert not missing_required_columns, f"DataFrame is missing required columns: {missing_required_columns}"
    extra_columns = _df_columns - _required_columns - _accepted_extra_columns
    assert not extra_columns, (f"DataFrame contains unrecognized columns: {extra_columns}. "
                               f"Accepted additional columns are: {_accepted_extra_columns}")

    nan_row_frac = _count_rows_with_nans(df) / df.shape[0]
    if nan_row_frac == 0:
        logger.debug(f"no NaNs in Dataframe.")

    if nan_row_frac != 0 and nan_row_frac < nan_threshold:
        logger.warning(f"DataFrame contains {nan_row_frac * 100:.1e}% rows with NaN values."
                       f"No raise, since below threshold: {nan_threshold * 100:.1e}%")
    if nan_row_frac > nan_threshold:
        raise UserWarning(f"DataFrame contains {_count_rows_with_nans(df)}/{df.shape[0]} "
                          f"= {nan_row_frac * 100:.2f}% of NaN rows.")


def _count_rows_with_nans(df: pd.DataFrame):
    """Count the number of rows containing NaN values in a DataFrame."""
    return (df.isna().sum(axis=1) > 0).sum()


def extract_segments(x_df: pd.DataFrame, column: str, condition: bool, time_unit: str = 'h',
                     add_during_night_indicator: bool = False, night_start: int = None, night_end: int = None,
                     during_night_h_thres: int = None):
    """ Find segments where a given binary column of a DataFrame is True/False.
    Attention: Ignores segments of length==1 and calculates the time between first and last time step,
    so not recommended for short segments + low sample rates.
    Parameters:
        :param x_df: (pd.DataFrame) The input data, must contain a bool column named like the given 'column'.
        :param column: (str) The indicator for the binary column to investigate.
        :param condition: (bool) Indicates if segments of True or False are returned.
        :param time_unit: (str) If 's', length of segment is given in seconds, if 'h' in hours. Defaults to hours.
        :param add_during_night_indicator: (bool) Whether to check if segment is during night or not. Default is False.
        :param night_start: (int, optional) The start hour to consider as night if 'add_during_night_indicator' is True.
        :param night_end: (int, optional) The end hour to consider as night if 'add_during_night_indicator' is True.
        :param during_night_h_thres (int, optional) If given, segment will only be considered as 'during_night'
                if overlap is larger than the given value (in hours).
    Returns:
        :return: (pd.DataFrame). One row per segment, containing 'start_time', 'end_time', 'length' and optionally \
        'during_night' if 'add_during_night_indicator' is True."""
    if add_during_night_indicator and (night_start is None or night_end is None):
        raise ValueError("'night_start' and 'night_end' must be provided if 'add_during_night_indicator' is set True.")
    assert time_unit in ('s', 'h'), f"'time_unit' param must be in 's' (seconds) or 'h' (hours), got {time_unit}."
    segment_id = (x_df[column] != x_df[column].shift(1)).cumsum()
    condition_segments = x_df[(x_df[column] if condition else ~x_df[column])]
    _segments_df = condition_segments.groupby(segment_id[condition_segments.index]).apply(lambda x: pd.Series({
        'start_time': x.index[0], 'end_time': x.index[-1],
        f'length({time_unit})': (x.index[-1] - x.index[0]) / pd.Timedelta(1, unit=time_unit)})).reset_index(drop=True)
    if add_during_night_indicator and not _segments_df.empty:
        assert during_night_h_thres is not None, \
            f"if 'add_during_night_indicator', 'during_night_h_thres' cannot be None."
        # set 'during_night' True if at least 'during_night_h_thres' hours are during pre-defined night window.
        _segments_df['during_night'] = _segments_df.apply(
            lambda row: _calculate_night_overlap_h(row['start_time'], row['end_time'], night_start, night_end, column)
                        >= float(during_night_h_thres), axis=1)
    return _segments_df


def _calculate_night_overlap_h(_start_time: pd.Timestamp, _end_time: pd.Timestamp, _night_start_hour: int,
                               _night_end_hour: int, column: str):
    """ Calculate the total hours of overlap between a time segment and defined night periods."""
    overlap_duration_h = 0
    n_midnights = (_end_time.normalize() - _start_time.normalize()).days + 1
    midnights = [(_start_time.normalize() + pd.Timedelta(days=i)) for i in range(n_midnights)]
    nights = []
    if n_midnights == 1:  # start and end are on the same day
        for midnight in midnights:
            for days_start, days_end in zip((-1, 0), (0, 1)):
                night_start_ = midnight + pd.Timedelta(days=days_start, hours=_night_start_hour)
                night_end_ = midnight + pd.Timedelta(days=days_end, hours=_night_end_hour)
                nights.append({'start': night_start_, 'end': night_end_})
                del night_start_
                del night_end_
    elif n_midnights == 2:
        night_start_ = midnights[0] + pd.Timedelta(days=0, hours=_night_start_hour)
        night_end_ = midnights[0] + pd.Timedelta(days=1, hours=_night_end_hour)
        nights.append({'start': night_start_, 'end': night_end_})
    else:
        if column == 'sptw':
            logger.warning(f"segment with start={_start_time} and end={_end_time} spans over several days,"
                           f"so it's likely misclassified.")
    for possible_night in nights:
        overlap_start = max(_start_time, possible_night['start'])
        overlap_end = min(_end_time, possible_night['end'])
        if overlap_start < overlap_end:
            overlap_duration_h += (overlap_end - overlap_start).total_seconds() / (60 * 60)
    return overlap_duration_h


def handle_duplicate_timestamps(df: pd.DataFrame, remove_dupes: bool = True, **print_info):
    """Handle duplicate timestamps in a DataFrame.
    Parameters:
        :param df: (pd.DataFrame): DataFrame with a DatetimeIndex.
        :param remove_dupes: (bool): Whether to remove duplicate timestamps. Defaults to True.
        :para,s **print_info: (dict) Optional information for logging (e.g., patient_id, sample_rate).
    Returns:
        :return: (pd.DataFrame) with duplicates removed or unchanged."""
    assert isinstance(df.index, pd.DatetimeIndex), "DataFrame must have 'DatetimeIndex"
    _n_dupes = (df.index.duplicated()).sum()
    if _n_dupes and remove_dupes:
        df = df[~df.index.duplicated(keep=False)]
        logger.info(f"(io: {print_info.get('patient_id')}) found {_n_dupes} duplicate timestamps."
                    f"removing ~{_n_dupes / print_info.get('sample_rate')}s of data" if print_info
                    else f"(io) found {_n_dupes} duplicate timestamps that will be removed.")
    elif _n_dupes and not remove_dupes:
        logger.warning(f"(io: {print_info.get('patient_id')})found {_n_dupes} duplicate timestamps."
                       f"Set 'resolve_duplicates' to resolve."
                       if print_info else f"[WARN]: (io) found {_n_dupes} duplicate timestamps."
                                          f"Set 'remove_dupes' to resolve.")
    else:
        logger.info(f"(io: {print_info.get('patient_id')}) no duplicate timestamps found." if print_info else
                    f"(io) no duplicate timestamps found.")
    return df


def infer_mean_sample_rate(df: pd.DataFrame, mute_logging: bool = True):
    """ Infer the mean and standard deviation of the sampling rate in a DataFrame.
    Parameters:
        :param df: (pd.DataFrame) Input data, must have a DatetimeIndex.
        :param mute_logging: (bool, Optional) If True, suppresses logging messages. Defaults to True.
    Returns:
        :return: (Tuple[bool, float, float])
            is_uniform (bool): Indicates whether the sampling rate is uniform.
            mean_sample_rate (float): Mean sampling rate in Hz, rounded to two decimals.
            std_sample_rate (float): Standard deviation of the sampling rate in Hz, rounded to two decimals."""
    assert isinstance(df.index, pd.DatetimeIndex), "DataFrame must have 'DatetimeIndex'"
    time_diffs = df.index.to_series().diff().dt.total_seconds().values[1:]
    with np.errstate(divide='ignore', invalid='ignore'):  # ignore warning when df still contains duplicate timestamps
        mean_sample_rate, std_sample_rate = 1 / np.mean(time_diffs), np.std(1 / time_diffs)  # not a typo
    is_uniform = True if len(np.unique(time_diffs)) == 1 and np.isclose(std_sample_rate, 0) else False
    if not is_uniform and not mute_logging:
        logger.info(f"non-uniform sampling rate detected: "
                    f"{len(np.unique(time_diffs))} unique values with "
                    f"mean fs = {mean_sample_rate:.2f} ± {std_sample_rate:.2f} Hz")
    elif len(np.unique(time_diffs)) == 1 and not mute_logging:
        logger.info(f"uniform sampling rate detected: fs = {mean_sample_rate:.2f} Hz")
    return is_uniform, np.round(mean_sample_rate, 2), np.round(std_sample_rate, 2)


def mmap_like(data, filename, mode='w+', shape=None):
    """ Create a memory-mapped NumPy array matching the DataFrame's index and columns."""
    dtype = np.dtype([(data.index.name, data.index.dtype), *[(c, data[c].dtype) for c in data.columns]])
    shape = shape or (len(data),)
    data_mmap = np.memmap(filename, dtype=dtype, mode=mode, shape=shape)
    return data_mmap


def copy2mmap(data: pd.DataFrame, data_mmap: np.memmap, flush: bool = True):
    """ Copy a pandas.DataFrame to a numpy.memmap. This operation is in-place."""
    data_mmap[data.index.name] = data.index.to_numpy()
    for c in data.columns:
        data_mmap[c] = data[c].to_numpy()
    if flush:
        np.memmap.flush(data_mmap)
    return


def mmap2df(data_mmap, index_col='time', copy=True):
    """ Convert a numpy structured array to pandas dataframe. """
    columns = [c for c in data_mmap.dtype.names if c != index_col]
    return pd.DataFrame({c: np.asarray(data_mmap[c]) for c in columns},
                        copy=copy, index=pd.Index(np.asarray(data_mmap[index_col]), name=index_col, copy=copy), )
