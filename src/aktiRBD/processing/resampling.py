import logging

import numpy as np
import pandas as pd

__all__ = ['resample_df_uniform']

logger = logging.getLogger(__name__)


def resample_df_uniform(data: pd.DataFrame, target_rate: int, interp_kwargs: dict = None, chunksize: int = None):
    """ Resample a given datetime indexed DataFrame to have a uniform sampling rate.
    Parameters:
        :param data: (pd.DataFrame) the input dataframe, must have a datetime index.
        :param target_rate: (int) the target sample rate in Hz.
        :param interp_kwargs (dict) kwargs to be passed to pd.DataFrame.interpolate().
        :param chunksize: (int) if not None, the size of chunks to be resampled at once. Else the whole df will be
            processed at once. (currently not supported since memory usage was handleable.)
    Returns:
        :return: (pd.DataFrame) The uniformly resampled dataframe. The quality control is handled outside the function,
        e.g. see Basedevice._resample_uniform"""
    if interp_kwargs is None:
        interp_kwargs = {'method': 'nearest'}
    elif interp_kwargs['method'] == 'spline':
        raise UserWarning(f"do not use 'spline' (https://github.com/pandas-dev/pandas/issues/26309)")

    assert isinstance(data.index, pd.DatetimeIndex), "DataFrame must have 'DatetimeIndex'"
    logger.debug(f"uniform resampling to {target_rate} Hz using {interp_kwargs}")
    _start, _end = data.index[0], data.index[-1]
    _n_new_ticks = np.round((_end - _start).total_seconds() * target_rate, 0).astype('int')

    if chunksize is None:
        _old_idx = data.index
        _new_idx = pd.date_range(_old_idx[0], _old_idx[-1], freq=f'{1 / target_rate:4f}s')
        data = (data.reindex(_old_idx.union(_new_idx))  # merge old and new indices
                .interpolate(**interp_kwargs)  # fill missing values by interpolation
                .reindex(_new_idx))  # extract only the wanted indices
        # todo: this is quite slow, speedup with parallel processing, numba, etc.?
        data.index.name = 'time'

    elif chunksize > 0:
        # challenge: if processed in chunks, how to ensure continuity of resampled chunks at the edges?
        raise NotImplementedError(f"chunk-wise re-sampling currently not supported, please use 'chunksize'=None.")

    return data
