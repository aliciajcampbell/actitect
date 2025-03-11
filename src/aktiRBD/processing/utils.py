import logging

import numpy as np
import pandas as pd

__all__ = ['assert_valid_df', 'extract_segments', 'handle_duplicate_timestamps', 'infer_mean_sample_rate',
           'mmap_like', 'mmap2df', 'copy2mmap']

logger = logging.getLogger(__name__)


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
                           f" so it's likely misclassified.")
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
                    f" removing ~{_n_dupes / print_info.get('sample_rate')}s of data" if print_info
                    else f"(io) found {_n_dupes} duplicate timestamps that will be removed.")
    elif _n_dupes and not remove_dupes:
        logger.warning(f"(io: {print_info.get('patient_id')})found {_n_dupes} duplicate timestamps."
                       f" Set 'resolve_duplicates' to resolve."
                       if print_info else f"[WARN]: (io) found {_n_dupes} duplicate timestamps."
                                          f" Set 'remove_dupes' to resolve.")
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
