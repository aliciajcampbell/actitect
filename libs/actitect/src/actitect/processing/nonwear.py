from dataclasses import dataclass

import pandas as pd

__all__ = ['segment_non_wear_episodes']


@dataclass(frozen=True)
class NonWearParams:
    min_duration: str = '60m'  # minimum duration of a stationary episode to be considered as non-wear
    window_size: str = '10s'  # rolling window size for stationary period detection
    stationary_std_tolerance: float = .015  # std. tolerance (in g) below which a window is considered stationary


def segment_non_wear_episodes(data: pd.DataFrame, nw_params=NonWearParams()):
    """ Detect nonwear episodes based on long periods of stationary recordings.
    Parameters:
           :param data: (pd.DataFrame)  containing the acceleration time-series. It must contain at least columns
            `x,y,z` and the index must be a DateTimeIndex.
        :param nw_params: (NonWearParams) see NonWearParams dataclass for details.
    Returns:
        :return: (Tuple[DataFrame, Dict]) containing the updated input df with an added boolean 'wear' column flagging
            the non-wear episodes and a dict containing summarizing stats."""

    info = {}
    stationary_indicator = (  # apply rolling window and detect stationary periods
            data['x'].resample(nw_params.window_size, origin='start').std().lt(nw_params.stationary_std_tolerance)
            & data['y'].resample(nw_params.window_size, origin='start').std().lt(nw_params.stationary_std_tolerance)
            & data['z'].resample(nw_params.window_size, origin='start').std().lt(nw_params.stationary_std_tolerance))

    # segment the start/end points of stationary periods:
    segment_edges = pd.Series(stationary_indicator != stationary_indicator.shift(1))
    segment_edges.iloc[0] = True  # first edge is always True
    segment_ids = segment_edges.cumsum()
    stationary_segment_ids = segment_ids[stationary_indicator]
    stationary_segment_lengths = (
        stationary_segment_ids.groupby(stationary_segment_ids)  # get segments grouped by id
        .agg(start_time=lambda x: x.index[0], length=lambda x: x.index[-1] - x.index[0])  # get segment start/length
        .set_index('start_time').squeeze(axis=1).astype('timedelta64[ns]')  # formatting
    )
    # filter for segments of relevant length (> min_nw_duration)
    nonwear_segment_lengths = \
        stationary_segment_lengths[stationary_segment_lengths > pd.Timedelta(nw_params.min_duration)]
    count_nonwear = len(nonwear_segment_lengths)
    total_nonwear = nonwear_segment_lengths.sum().total_seconds()
    total_wear = \
        data.index.to_series().diff().pipe(lambda x: x[x < pd.Timedelta('1s')].sum()).total_seconds() - total_nonwear

    info.update({'total_wear_time(d)': total_wear / (60 * 60 * 24), 'non_wear_time(d)': total_nonwear / (60 * 60 * 24),
                 'num_non_wear_episodes': count_nonwear})
    data['wear'] = True  # create wear mask:
    for start_time, length in nonwear_segment_lengths.items():
        data.loc[start_time:start_time + length, 'wear'] = False

    return data, info
