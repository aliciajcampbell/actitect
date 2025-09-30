import datetime
from contextlib import contextmanager

import numpy as np
import pandas as pd

from actitect.utils import df_astype_inplace

__all__ = ['segment_nocturnal_movements']


def segment_nocturnal_movements(x_df: pd.DataFrame, selected_sptws: pd.DataFrame):
    """ Identify movement segments in the sleep actimeter recordings based on thresholding the mean acceleration.
    Parameters:
        :param x_df: (pd.DataFrame) containing the entire recording of one patient/study. Must have a datetime index and
            contain the columns 'x', 'y', and 'z' (each in g).
        :param selected_sptws: (pd.DataFrame) indicating the identified sleep period time windows in which to search
            for the movement bouts. Has to include columns 'start_time' and 'end_time'.
    Returns:
        :return: (pd.DataFrame) The original DataFrame with an inplace update boolean column 'move'. """

    x_df.insert(3, 'mag', np.linalg.norm([x_df.x, x_df.y, x_df.z], axis=0))

    mag_mean_sleep = np.mean(x_df.loc[x_df['sptw'] & x_df['wear'], 'mag'])
    mag_std_sleep = np.std(x_df.loc[x_df['sptw'] & x_df['wear'], 'mag'])
    move_threshold = max(mag_mean_sleep + mag_std_sleep, .1)

    # choose only selected sptws to filter relevant movements
    time_window_mask = np.zeros(len(x_df), dtype=bool)
    for _, window in selected_sptws.iterrows():
        time_window_mask |= (x_df.index >= window['start_time']) & (x_df.index <= window['end_time'])

    # check for movements in sleep (= simply defined as current acceleration > mean acceleration ± std)
    # if True, set movement to True, otherwise NaN (not False, see ffill(), bfill())
    x_df['movement'] = \
        np.where((x_df['mag'] > move_threshold) & (x_df['sptw'] & x_df['wear'] & time_window_mask), 1, np.NaN,)
    # calculate time difference between movements=True:
    with time_to_integer_index(x_df):
        time_diff = x_df.loc[x_df.movement == 1, 'time'].diff()
        if len(time_diff) > 1:
            # no movement before/after first/last detected movement:
            if time_diff.index[0] - 1 in x_df.index and time_diff.index[-1] + 1 in x_df.index:
                x_df.loc[time_diff.index[0] - 1, 'movement'] = 0  # reason for context manager
                x_df.loc[time_diff.index[-1] + 1, 'movement'] = 0
            else:
                raise ValueError("Index out of range: Required indices not found in DataFrame")

            if time_diff.index[-1] + 1 in x_df.index:
                x_df.loc[time_diff.index[-1] + 1, 'movement'] = 0
            # if movements are seperated more than 1s, separate them with 'False' seperator
            x_df.loc[time_diff[time_diff > datetime.timedelta(seconds=1)].index - 1, 'movement'] = 0
        else:
            x_df['movement'] = np.NaN
    # backward and forward fill NaN values to separate the segments:
    x_df.movement = x_df.movement.bfill().ffill()
    df_astype_inplace(x_df, {'movement': bool})
    # make binary that assigns unique ID to every segment of consecutive movement=True/False
    segment_mask = (x_df['movement'] != x_df['movement'].shift()).cumsum()
    # select only the movement periods (even):
    move_segment_ids = segment_mask.unique()[1::2]
    stats = {'mag_mean_sleep': np.round(mag_mean_sleep, 4),
             'mag_std_sleep': np.round(mag_std_sleep, 4),
             'move_threshold': np.round(move_threshold, 4)}
    return segment_mask, move_segment_ids, stats


@contextmanager
def time_to_integer_index(df):
    try:
        df.reset_index(inplace=True)
        yield df
    finally:
        df.set_index('time', inplace=True)
