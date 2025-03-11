import logging
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from aktiRBD.utils import dict_minus_key

logger = logging.getLogger(__name__)

__all__ = ['SleepDetector']


@dataclass(frozen=True)
class SleepParams:
    day_offset: int = 12  # offset from midnight for start of days when grouping by day (hours).
    raw_epoch_length: int = 5  # length of epoch for rolling median and z angle averages (HDCZA steps 1-3) (seconds).
    z_epoch_length: int = 375  # length of epoch for rolling median of z angle change (HDCZA step 5) (seconds).
    min_wear_hours: int = 3  # min. duration of wear during day to proceed sleep detection (hours).
    min_sptw_length: int = 30  # minimum length of sptw (minutes).
    max_gap_time: int = 60  # maximum gap between sptw to be considered continuous (minutes).
    z_per_threshold: int = 10  # percentile threshold of z-angle change used in sptw detection (percentile).
    z_abs_threshold: int = 5  # absolute threshold of z-angle change used for sleep-bout detection (degree).
    min_sleep_length: int = 5  # min. duration for sleep bouts (minutes).
    max_sptw_length_h: int = 16  # max. length for sptws (hours).
    overnight_start: int = 22  # start point for overnight indicator. (hours)
    overnight_end: int = 8  # end point for overnight indicator. (hours)


class SleepDetector:
    """ A class to detect sleep period time windows (sptws) and sleep bouts from accelerometer and temperature data
    using the HDCZA algorithm as described in van Hees et al., 2018. Sptws are broader windows in which sleep likely
    occurs, marked by the first sleep onset and the last sleep episode of the night. Sleep bouts are segments within
    sptws where sleep occurs with higher confidence.
    References:
    - van Hees, E. M., Brouwer, A., & Nieuwenhuijsen, K. (2018). A heuristic algorithm looking at distribution of
      change in z-angle (HDCZA) to assess physical activity intensity using wrist-worn accelerometers.
      Scientific Reports, 8(1), 11822. https://www.nature.com/articles/s41598-018-31266-z
    - Beyer, K.B., Weber, K.S., Cornish, B.F. et al. NiMBaLWear analytics pipeline for wearable sensors: a modular,
      open-source platform for evaluating multiple domains of health and behaviour.
      BMC Digit Health 2, 8 (2024). https://doi.org/10.1186/s44247-024-00062-3 """
    def __init__(self, sleep_params: Optional[SleepParams] = None): 
        self.sleep_params = sleep_params if sleep_params else SleepParams()
        self.sample_rate = None

    def fit(self, data: pd.DataFrame, sample_rate: int, include_sleep_bouts: bool = True):
        self.sample_rate = sample_rate
        # apply HDCZA to detect the sleep period time windows (sptws) and sleep bouts (if include_sleep_bouts)
        sptws, z_angle_df = self._detect_sptws(data)
        sleep_bouts = self._detect_sleep_bouts(z_angle_df, sptws, data.index[0]) if include_sleep_bouts else None
        # summarize some stats and format output
        _, info = self._get_sleep_summary(sptws, sleep_bouts)
        return self._annotate_df(data, sptws, sleep_bouts), info, sptws

    def _detect_sptws(self, data: pd.DataFrame):
        """ Main logic to detect sleep-period time windows (sptws). Implements HDCZA steps 1-8.
            -step1-4: see 'self._z_angle_change'
            -step5: rolling median of z-angle change (default window is 5min, see SleepParams)
            -step6: apply percentile threshold to detect stationary segments. (default 10perc., see SleepParams)
            -step7+8: drop short sptws and merge close ones. (defaults is short < 30min, and merge < 60min gap)
        Parameters:
            :param data: (pd.DataFrame) containing tri-axial accelerometric data in columns 'x','y' and 'z' (g).
        Returns:
            :return: (Tuple[pd.DataFrame]) containing the extract±es sptws and the used z_angle for further analysis."""

        # step1-5:
        z_angle_df = self._z_angle_change(data, n_epoch_samples=self.sample_rate * self.sleep_params.raw_epoch_length)
        window_size = round(self.sleep_params.z_epoch_length / self.sleep_params.raw_epoch_length)
        z_angle_diff_med = pd.Series(z_angle_df.z_angle_change).rolling(window_size).median()[window_size - 1:]
        z_sample_rate = 1 / self.sleep_params.raw_epoch_length

        start_datetime = data.index[0]
        sptws = pd.DataFrame()
        for start_index, z_day in self._slice_data_by_day(  # loop over each day
                z_angle_diff_med, start_datetime, self.sleep_params.day_offset, z_sample_rate):

            z_day_wear = z_day[z_day >= 0]
            if len(z_day_wear) > (self.sleep_params.min_wear_hours * 60 * 60 * z_sample_rate):
                # step 6
                z_angle_threshold = np.percentile(a=z_day_wear, q=self.sleep_params.z_per_threshold) * 15
                z_below_per_threshold = ((z_day >= 0) & (z_day < z_angle_threshold)).tolist()
                z_angle_cross_threshold = [j for j in range(1, len(z_below_per_threshold))
                                           if z_below_per_threshold[j] != z_below_per_threshold[j - 1]]
                if z_angle_cross_threshold:
                    if not z_below_per_threshold[z_angle_cross_threshold[0]]:
                        z_angle_cross_threshold.insert(0, 0)
                    if z_below_per_threshold[z_angle_cross_threshold[-1]]:
                        z_angle_cross_threshold.append(len(z_below_per_threshold) - 1)
                    sptw_candidates = self._get_candidates(z_angle_cross_threshold, z_data=z_day, mode='sptw')
                    _day_sptws = pd.DataFrame({'start_time': [start_datetime + pd.Timedelta(
                        seconds=((start_index + j) * self.sleep_params.raw_epoch_length)) for j in sptw_candidates[0]],
                                               'end_time': [start_datetime + pd.Timedelta(
                                                   seconds=((start_index + j) * self.sleep_params.raw_epoch_length))
                                                            for j in sptw_candidates[1]]})
                    if not _day_sptws.empty:
                        sptws = pd.concat([sptws, _day_sptws], ignore_index=True)

        if not sptws.empty:
            sptws = self._format_sptws(sptws, self.sleep_params)
        return sptws, z_angle_df

    def _detect_sleep_bouts(self, z_angle_df: pd.DataFrame, sptws: pd.DataFrame, start_datetime: pd.Timestamp):
        """ Detect sleep bout segments within each sptw. A sleep bout is a segment of higher confidence for sleep, i.e.
        a lower z-angle change threshold.
        Parameters:
            :param z_angle_df: (pd.Dataframe) containing the z-angle rolling average in 'z_angle_avg' and its abs.
                change over time in 'z_angle_change'.
            :param sptws: (pd.DataFrame) containing the relevant sptws.
            :param start_datetime: (pd.Datetime) indicating the first recorded timestamp.
        Return:
            :return: (pd.DataFrame) with row per sleep bout with information about number and start/end time."""
        sptws = sptws.copy()
        z_sample_rate = 1 / self.sleep_params.raw_epoch_length
        sleep_bouts = pd.DataFrame()
        for _sptw_num, _sptw in sptws.iterrows():
            start_index = round((_sptw['start_time'] - start_datetime).total_seconds() * z_sample_rate)
            end_index = round((_sptw['end_time'] - start_datetime).total_seconds() * z_sample_rate)
            z_angle_diff = np.array(z_angle_df.z_angle_change)
            z_below_per_threshold = (z_angle_diff[start_index:end_index] <= self.sleep_params.z_abs_threshold).tolist()
            z_angle_cross_threshold = [j for j in range(1, len(z_below_per_threshold))  # keep blocks > min_sleep_length
                                       if z_below_per_threshold[j] != z_below_per_threshold[j - 1]]
            if all(z_below_per_threshold):  # if all values below threshold then crossed threshold on first index
                z_angle_cross_threshold.insert(0, 0)

            if z_angle_cross_threshold:
                if not z_below_per_threshold[z_angle_cross_threshold[0]]:
                    z_angle_cross_threshold.insert(0, 0)
                if z_below_per_threshold[z_angle_cross_threshold[-1]]:
                    z_angle_cross_threshold.append(len(z_below_per_threshold) - 1)

                sleep_candidates = self._get_candidates(z_angle_cross_threshold, mode='sleep_bout')
                if sleep_candidates[0]:
                    _sptw_sleep_bouts = pd.DataFrame({
                        'sptw_num': [_sptw_num] * len(sleep_candidates[0]),
                        'start_time': [start_datetime + pd.Timedelta(
                            seconds=((start_index + j) * self.sleep_params.raw_epoch_length)) for j in
                                       sleep_candidates[0]],
                        'end_time': [start_datetime + pd.Timedelta(
                            seconds=((start_index + j) * self.sleep_params.raw_epoch_length)) for j in
                                     sleep_candidates[1]],
                        'overnight': [_sptw['overnight']] * len(sleep_candidates[0])
                    })
                    if not _sptw_sleep_bouts.empty:
                        sleep_bouts = pd.concat([sleep_bouts, _sptw_sleep_bouts], ignore_index=True)

        sleep_bouts['sleep_bout_num'] = list(range(sleep_bouts.shape[0]))
        return sleep_bouts

    def _annotate_df(self, df_in: pd.DataFrame, sptws: pd.DataFrame, sleep_bouts: pd.DataFrame):
        """Adds boolean 'sptw' 'sleep_bout' (if not None) columns to original DataFrame to indicate sleep segments."""
        # Initialize columns
        df_in['sptw'] = False
        if isinstance(sleep_bouts, pd.DataFrame) and not sleep_bouts.empty:
            df_in['sleep_bout'] = False
        df_index = df_in.index.values
        sptw_start = sptws['start_time'].values
        sptw_end = sptws['end_time'].values
        valid_sptw = sptws['length(h)'].values <= self.sleep_params.max_sptw_length_h
        for start, end in zip(sptw_start[valid_sptw], sptw_end[valid_sptw]):
            sptw_mask = (df_index >= start) & (df_index <= end)
            df_in['sptw'] |= sptw_mask
        if isinstance(sleep_bouts, pd.DataFrame) and not sleep_bouts.empty:
            sleep_bout_start = sleep_bouts['start_time'].values
            sleep_bout_end = sleep_bouts['end_time'].values

            for start, end in zip(sleep_bout_start, sleep_bout_end):
                sleep_bout_mask = (df_index >= start) & (df_index <= end) & df_in['sptw'].values
                df_in['sleep_bout'] |= sleep_bout_mask

        return df_in

    def _get_candidates(self, z_angle_cross_thresh: np.ndarray, mode: str, z_data: np.ndarray = None):
        """ Pair and filter sptw/sleep candidates.  For sptws use ('mode'=sptw) or sleep_bouts ('mode'=sleep_bout)."""
        assert mode in ('sptw', 'sleep_bout'), f"'mode' must be either 'sptw' or 'sleep_bout', not '{mode}'"
        if mode == 'sptw':
            assert z_data is not None, f"'z_day' must be provided "
        short_thresh_min = self.sleep_params.min_sptw_length if mode == 'sptw' else self.sleep_params.min_sleep_length

        candidates = np.reshape(z_angle_cross_thresh, (math.floor(len(z_angle_cross_thresh) / 2), 2)).tolist()

        idx_too_short, j = [], 0
        for s in candidates:  # filter for duration < min_sptw_length (default 30mins) (HDCZA step7)
            if (s[1] - s[0]) < (short_thresh_min * 60 / self.sleep_params.raw_epoch_length):
                idx_too_short.append(j)
            j += 1
        candidates = np.delete(candidates, idx_too_short, 0).T.tolist()

        if mode == 'sptw':
            idx_merge = []
            for j in range(1, len(candidates[0])):  # filter for gap < max_gap_time(default 60mins)
                if (((candidates[0][j] - candidates[1][j - 1]) < (
                        self.sleep_params.max_gap_time * 60 / self.sleep_params.raw_epoch_length))
                        & (all(y >= 0 for y in z_data[candidates[1][j - 1]:candidates[0][j]]))):
                    idx_merge.append(j)
            for j in sorted(idx_merge, reverse=True):
                del candidates[0][j]
                del candidates[1][j - 1]

        return candidates

    @staticmethod
    def _format_sptws(sptws: pd.DataFrame, sleep_params: SleepParams):
        """Formats the detected sptws DataFrame and adds 'relative_date', 'duration' and 'overnight' columns."""
        sptws.insert(0, 'relative_date', [(x - pd.Timedelta(
            hours=sleep_params.day_offset)).date() for x in sptws['start_time']])
        sptws['sptw_num'] = list(range(len(sptws)))
        sptws.set_index('sptw_num', inplace=True)
        sptws['overnight'] = \
            ((sptws['start_time'] < (pd.to_datetime(sptws['relative_date']) + pd.Timedelta(
                days=1, hours=sleep_params.overnight_end)))
             & (sptws['end_time'] > (pd.to_datetime(sptws['relative_date']) + pd.Timedelta(
                        hours=sleep_params.overnight_start))))
        return sptws

    @staticmethod
    def _slice_data_by_day(z: pd.Series, start_datetime: pd.Timestamp, day_offset: int, z_sample_rate: int):
        """ Generator function that slices the input data by days.
        Parameters:
            :param z: (pd.Series) the data to slice.
            :param start_datetime: (pd.Timestamp) the first timestamp in data recordings.
            :param day_offset: (int) the offset to define when a day starts in hours.
                Default is defined in SleepParams as 12, i.e. noon-to-noon-
            :param z_sample_rate: (int) the sample rate for the data after the window averaging.
        Returns:
            :yield: (Tuple) of the data slice and start index for each day. """
        first_day_start = start_datetime.floor('d') + pd.Timedelta(hours=day_offset)
        days = ((len(z) / z_sample_rate) + (start_datetime - first_day_start).total_seconds()) / (
                60 * 60 * 24)
        day_start_times = [first_day_start + pd.Timedelta(days=x) for x in range(math.ceil(days))]
        day_start_times[0] = start_datetime
        day_start_indices = [round((day_start_time - start_datetime).total_seconds() * z_sample_rate) for day_start_time
                             in day_start_times]
        for i, start_index in enumerate(day_start_indices):
            end_index = day_start_indices[i + 1] if (i + 1) < len(day_start_indices) else None
            yield start_index, np.array(z[start_index:end_index])

    @staticmethod
    def _z_angle_change(data: pd.DataFrame, n_epoch_samples: int):
        """ Calculate consecutive 5sec average z-angle and its absolute change over time. (HDCZA steps 1-4).
            - step1: rolling median of raw signals x,y,z (default window is 5s, see SleepParams)
            - step2: z-angle of rolling medians.
            - step3: consecutive averages windows averages (default window is 5s, see SleepParams)
            - step4: absolute difference between successive values.
        Parameters:
            :param data: (DataFrame) accelerometer data x,y,z values (g) and with datetime index.
            :param n_epoch_samples: (int) number of samples per epoch.
        Returns:
            :return (DataFrame): containing the averages in 'z_angle_avg' (step3) and abs. changes
                in 'z_angle_change' (step4) with datetime index. """
        epoch_samples = int(n_epoch_samples)
        x_median, y_median, z_median = SleepDetector._compute_rolling_median(
            data.x, data.y, data.z, epoch_samples)  # step 1
        z_angle = np.degrees(np.arctan(z_median / np.hypot(x_median, y_median)))[epoch_samples - 1:]  # step 2
        z_angle_avg = [np.mean(z_angle[i:i + epoch_samples]) for i in range(0, len(z_angle), epoch_samples)]  # step 3
        z_angle_change = np.insert(np.abs(np.diff(z_angle_avg)), 0, 0)  # step 4
        epoch_timestamps = \
            data.index[epoch_samples - 1: epoch_samples - 1 + len(z_angle_avg) * epoch_samples: epoch_samples]
        return pd.DataFrame({'z_angle_avg': z_angle_avg, 'z_angle_change': z_angle_change}, index=epoch_timestamps)

    @staticmethod
    def _compute_rolling_median(x, y, z, window_size, n_jobs: int = -1):
        """ Compute rolling medians for x,y,z series in parallel.
        Parameters:
            :param x: (array-like) Data series for the x-axis.
            :param y: (array-like) Data series for the y-axis.
            :param z: (array-like) Data series for the z-axis.
            :param window_size: (int) The sample size of the rolling window.
            :param n_jobs: (int, Optional) How many cores for parallel computation.
                Default is -1 to use all available.
        Returns:
            :return: (tuple) containing rolling medians for X, Y, and Z as Pandas Series.
        """
        medians = Parallel(n_jobs=n_jobs)(
            delayed(lambda s: pd.Series(s).rolling(window=window_size).median())(series)
            for series in [x, y, z])
        return [med_series.to_numpy() for med_series in medians]

    def _get_sleep_summary(self, sptws: pd.DataFrame, sleep_bouts: pd.DataFrame):
        """ Summarizes the detected sptws and sleep bouts. Returns a Tuple of sptws with updated columns and an info
        with summarizing statistics."""

        info = {'num_sptws': sptws.shape[0]}
        if isinstance(sleep_bouts, pd.DataFrame) and not sleep_bouts.empty:
            stats = self.sptw_stats(sptws, sleep_bouts)
            sptws['duration(h)'] = np.round(stats['sptw_duration'].values, 4)
            sptws['sleep(h)'] = np.round(stats['sleep_duration'].values, 4)
            sptws['se'] = np.round(stats['se'].values, 4)
            sptws['waso'] = np.round(stats['waso'].values, 4)
            info.update({'num_sleep_bouts': sleep_bouts.shape[0]})

        sptw_dict = dict_minus_key(sptws.to_dict(), ['relative_date', 'sptw_num'])
        info.update({'sptws': {index: {key: value_list[index] for key, value_list in sptw_dict.items()
                                       if index in value_list} for index in sptw_dict['start_time'].keys()}})
        sptws['length(h)'] = \
            (pd.to_datetime(sptws['end_time']) - pd.to_datetime(sptws['start_time'])) / pd.Timedelta(hours=1)
        return sptws, info

    @staticmethod
    def sptw_stats(sptws: pd.DataFrame, sleep_bouts: pd.DataFrame):
        """Calculate various sleep metrics for each sptw by analyzing associated sleep bouts.
        Parameters:
            :param sptws: (pd.DataFrame) containing sleep period time windows with columns:
            :param sleep_bouts: (pd.DataFrame) containing sleep bouts with columns:

        Returns:
            :return: (pd.DataFrame) indexed by 'sptw_num' containing the following columns:
                    - 'start_time': Start time of the sptw.
                    - 'end_time': End time of the sptw.
                    - 'type': Type of the sptw (set to 'sptw').
                    - 'sptw_duration': Duration of the sptw in hours.
                    - 'sleep_duration': Total sleep duration within the sptw in hours.
                    - 'sleep_to_wake_duration': Duration from first sleep onset to last wake time in hours.
                    - 'se': Sleep efficiency (sleep duration divided by sptw duration).
                    - 'waso': Wake After Sleep Onset within the sptw in hours."""
        sptw = sptws.copy()
        sptw['start_time'] = pd.to_datetime(sptw['start_time'], format='%Y-%m-%d %H:%M:%S')
        sptw['end_time'] = pd.to_datetime(sptw['end_time'], format='%Y-%m-%d %H:%M:%S')
        sptw['duration'] = [round((y - x).total_seconds() / 60) for (x, y) in zip(sptw['start_time'], sptw['end_time'])]
        sleep_bouts = sleep_bouts.copy()
        sleep_bouts['start_time'] = pd.to_datetime(sleep_bouts['start_time'], format='%Y-%m-%d %H:%M:%S')
        sleep_bouts['end_time'] = pd.to_datetime(sleep_bouts['end_time'], format='%Y-%m-%d %H:%M:%S')
        sleep_bouts['duration'] = [round((y - x).total_seconds() / 60)
                                   for (x, y) in zip(sleep_bouts['start_time'], sleep_bouts['end_time'])]

        sleep_stats = {'sptw_num': [], 'start_time': [], 'end_time': [], 'type': [], 'sptw_duration': [],
                       'sleep_duration': [], 'sleep_to_wake_duration': [], 'se': [], 'waso': []}

        for (cur_sptw_num, sptw_cur) in sptw.iterrows():  # loop through each sptw
            sleep_bouts_sptw = sleep_bouts.loc[sleep_bouts['sptw_num'] == cur_sptw_num]
            sleep_bouts_sptw.reset_index(inplace=True)
            sptw_duration = sptw_cur['duration']
            total_sleep_duration = sum(sleep_bouts_sptw['duration'])
            if not sleep_bouts_sptw.empty:
                sleep_onset_time = sleep_bouts_sptw.iloc[0]['start_time']
                waking_time = sleep_bouts_sptw.iloc[-1]['end_time']
                sleep_to_wake_duration = round((waking_time - sleep_onset_time).total_seconds() / 60)
            else:
                sleep_to_wake_duration = 0

            sleep_eff = round(total_sleep_duration / sptw_duration, 4) if sptw_duration != 0 else 0
            waso = sleep_to_wake_duration - total_sleep_duration
            sleep_stats['sptw_num'].append(cur_sptw_num)
            sleep_stats['start_time'].append(sptw_cur['start_time'])
            sleep_stats['end_time'].append(sptw_cur['end_time'])
            sleep_stats['type'].append('sptw')
            sleep_stats['sptw_duration'].append(sptw_duration / 60)
            sleep_stats['sleep_duration'].append(total_sleep_duration / 60)
            sleep_stats['sleep_to_wake_duration'].append(sleep_to_wake_duration / 60)
            sleep_stats['se'].append(sleep_eff)
            sleep_stats['waso'].append(waso / 60)

        return pd.DataFrame(sleep_stats).set_index('sptw_num')
