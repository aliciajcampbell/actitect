import logging
import os
import tempfile
from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm

from actitect.processing.utils import mmap_like, mmap2df, copy2mmap

__all__ = ['van_hees_sphere_calibration']

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CalibParams:
    calibration_cube_range_threshold: float = .3  # min. range (in g) the stat. points have to cover for calibration
    min_num_stat_samples: int = 50  # minimum n° of stationary samples needed for calibration
    window_size: str = '10s'  # rolling window size for stationary period detection
    stationary_std_tolerance: float = .015  # std. tolerance (in g) below which a window is considered stationary
    chunksize: int = 1_000_000  # n° of rows to process at once for memory efficiency, only use for calibrating input
    include_temperature: bool = False  # whether to calibrate temperature or not
    fit_max_iter: int = 1_000  # max. n° of iterations for calibration in case it does not converge
    improvement_tolerance: float = 1e-4  # min. relative improvement in calib. error required to continue iterations
    error_tolerance: float = .01  # max. acceptable calib. error to consider in g


def van_hees_sphere_calibration(data: pd.DataFrame, clb_params=CalibParams()):
    """ Auto calibration method based on unit sphere matching as described in van Hees et al.
    2014 (https://pubmed.ncbi.nlm.nih.gov/25103964/). Adapted from https://github.com/OxWearables/actipy/.
    Parameters:
        :param data: (pd.DataFrame)  containing the acceleration time-series. It must contain at least columns `x,y,z`
            and the index must be a DateTimeIndex.
        :param clb_params: (CalibParams) see CalibParams dataclass for details.
    Returns:
        :return: (Tuple[DataFrame, Dict]) containing the updated input df with calibrated data and dict summarizing
            calibration process. """
    info = {}
    stationary_indicator = (  # apply rolling window and detect stationary periods
            data['x'].resample(clb_params.window_size, origin='start').std().lt(clb_params.stationary_std_tolerance)
            & data['y'].resample(clb_params.window_size, origin='start').std().lt(clb_params.stationary_std_tolerance)
            & data['z'].resample(clb_params.window_size, origin='start').std().lt(clb_params.stationary_std_tolerance))

    sphere = data[['x', 'y', 'z']].resample(  # sphere of the stationary periods
        clb_params.window_size, origin='start').mean()[stationary_indicator].dropna().to_numpy()
    nonzero = np.linalg.norm(sphere, axis=1) > 1e-8
    sphere = sphere[nonzero]  # remove zero's

    _use_temperature = 'temperature' in data and clb_params.include_temperature
    if _use_temperature:
        temperature = data['temperature'].resample(
            clb_params.window_size, origin='start').mean()[stationary_indicator].dropna().to_numpy()
        temperature = temperature[nonzero]
    del stationary_indicator
    del nonzero

    if len(sphere) < clb_params.min_num_stat_samples:  # check if sufficient n° of stationary samples for calibration
        info['calib_ok'] = 0
        info['calib_error_before(mg)'], info['calib_error_before(mg)'] = np.nan, np.nan
        logger.warning(f"skipping calibration: insufficient number of stationary samples found:"
                       f"{len(sphere)} < {clb_params.min_num_stat_samples}")
        return data, info

    offset = np.array([0.0, 0.0, 0.0], dtype=sphere.dtype)
    gain = np.array([1.0, 1.0, 1.0], dtype=sphere.dtype)
    best_offset, best_gain = np.copy(offset), np.copy(gain)
    if _use_temperature:
        gain_temperature = np.array([0.0, 0.0, 0.0], dtype=temperature.dtype)
        best_gain_temperature = np.copy(gain_temperature)

    # calculate deviation between current uncalibrated ellipsoid and target unit sphere
    curr = sphere
    target = curr / np.linalg.norm(curr, axis=1, keepdims=True)
    errors = np.linalg.norm(curr - target, axis=1)
    err = np.mean(errors)  # MAE more robust to outliers than RMSE which is used in paper
    init_err, best_err = err, 1e16  # init best err with large number
    info['calib_error_before(mg)'] = init_err * 1000

    # check if stationary points are sufficiently distributed within calibration cube range along all axes
    if (np.max(sphere, axis=0) < clb_params.calibration_cube_range_threshold).any() \
            or (np.min(sphere, axis=0) > -clb_params.calibration_cube_range_threshold).any():
        info['calib_ok'] = 0
        info['calib_error_after(mg)'] = init_err * 1000
        logger.warning(f"skipping calibration: insufficient number of uniformly distributed stationary points.")
        return data, info

    for iteration in range(clb_params.fit_max_iter):  # iterative closest point fitting process
        # calc. the weights, quantile reduce sensibility to outliers
        _max_err = np.quantile(errors, .995)
        weights = np.maximum(1 - errors / _max_err, 0)
        for k in range(3):  # optimize params for each acceleration axis
            _in, _out = curr[:, k], target[:, k]
            if _use_temperature:
                _in = np.column_stack((_in, temperature))
            _in = sm.add_constant(_in, prepend=True, has_constant='add')
            fit_params = sm.WLS(_out, _in, weights=weights).fit().params
            offset[k] = fit_params[0] + (offset[k] * fit_params[1])
            gain[k] = fit_params[1] * gain[k]
            if _use_temperature:
                gain_temperature[k] = fit_params[2] + (gain_temperature[k] * fit_params[1])

        curr = offset + (sphere * gain)  # update the ellipsoid with current offset and gain
        if _use_temperature:
            curr = curr + (temperature[:, None] * gain_temperature)

        # update errors
        target = curr / np.linalg.norm(curr, axis=1, keepdims=True)
        errors = np.linalg.norm(curr - target, axis=1)
        err = np.mean(errors)
        err_improv = (best_err - err) / best_err
        if err < best_err:
            best_offset = np.copy(offset)
            best_gain = np.copy(gain)
            if _use_temperature:
                best_gain_temperature = np.copy(gain_temperature)
            best_err = err
        if err_improv < clb_params.improvement_tolerance:
            break  # improvement not significant, stop iteration

    info['calib_error_after(mg)'] = best_err * 1_000
    if (best_err > clb_params.error_tolerance) or (iteration + 1 == clb_params.fit_max_iter):
        info['calib_ok'] = 0  # either best_error still larger than tolerance or iterations > maximum n° of iterations
        return data, info

    else:  # calibration successful, update data
        with tempfile.TemporaryDirectory() as tmpdir:
            n_rows = len(data)
            mmap_fname = os.path.join(tmpdir, 'data.mmap')
            data_mmap = mmap_like(data, mmap_fname, shape=(n_rows,))
            for i in range(0, n_rows, clb_params.chunksize):

                current_chunksize = min(clb_params.chunksize, n_rows - i)
                # bug fix: after refactor, the global dataclass chunksize got updated,
                # since the new value was very large, huge computational increase over repetitions
                # (NEEDS TO BE VALIDATED!)

                chunk = data.iloc[i:i + current_chunksize]
                chunk_xyz = chunk[['x', 'y', 'z']].to_numpy()
                chunk_xyz = best_offset + best_gain * chunk_xyz
                if _use_temperature:
                    chunk_temperature = chunk['temperature'].to_numpy()
                    chunk_xyz = chunk_xyz + best_gain_temperature * chunk_temperature[:, None]
                chunk = chunk.copy(deep=True)  # copy to avoid modifying original data
                chunk[['x', 'y', 'z']] = chunk_xyz
                copy2mmap(chunk, data_mmap[i:i + current_chunksize])
            del data
            data = mmap2df(data_mmap, copy=True)
            del data_mmap

        info.update({'calib_ok': 1, 'calib_num_iter': iteration+1, 'calib_num_samples': len(sphere),
                     'calib_offset(xyz)': best_offset, 'calib_gain(xyz)': best_gain})
        if _use_temperature:
            info.update({'calib_gain_temp(xyz)': best_gain_temperature})

    return data, info
