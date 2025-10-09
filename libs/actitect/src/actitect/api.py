import logging
import os
from importlib.metadata import version as _pkg_version
from pathlib import Path
from typing import Optional, Union, Tuple, Dict

import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from . import utils
from .actimeter import ActimeterFactory
from .vis import draw_actigraphy_data
from .processing import filter_sptws, select_night_sptws, segment_nocturnal_movements
from .features import compute_per_night_sleep_features

logger = logging.getLogger(__name__)
utils.setup_logging()

__version__ = _pkg_version('actitect')

__all__ = [
    '__version__',
    'load',
    'process',
    'plot',
    'compute_sleep_motor_features'
]


def load(file_path: Union[str, os.PathLike, Path], subject_id: Optional[str] = None, *, header_only: bool = False) \
        -> pd.DataFrame:
    """ Load the raw data of a single actigraphy file (device auto-detected).
    Parameters
        :param file_path: (str | os.PathLike | Path) Path to the raw device file (e.g., Axivity .cwa, GENEActiv .bin,
            ActiGraph .gt3x, or supported CSV).
        :param subject_id: (str, optional) Identifier used in downstream metadata; if omitted, the placeholder 'none'
            is used.
        :param header_only : (bool, default False) If True, will only parse the header of the binary file, without
            loading the full data into memory.

    Returns
        :return: (Tuple[pd.DataFrame, Dict]) A time-indexed DataFrame containing the raw actigraphy data. The
            columns are time,x,y,z The df is empty if 'header_only'. And an info dict, containing either information
             about the raw data or just the parsed binary header if 'header_only'. """

    device = ActimeterFactory(file_path, patient_id=subject_id or 'none')
    if not header_only:
        df, info = device.load_raw_data(), device.get_info()
    else:
        df, info = pd.DataFrame(), device.load_raw_data(header_only=header_only)
    return df, info


def process(
        data: Union[str, os.PathLike, Path, pd.DataFrame],
        subject_id: Optional[str] = None,
        *,
        resample_rate: Union[int, str] = 'infer',
        highpass_hz: Optional[float] = .8,
        lowpass_hz: Optional[float] = 20,
        calibrate: bool = True,
        detect_nonwear: bool = True,
        detect_sleep: bool = True,
) -> Tuple[pd.DataFrame, Dict]:
    """ Load and preprocess a single actigraphy file (device auto-detected) by
        i. selecting the correct device reader via ActimeterFactory,
        ii. loading raw data,
        iii. applying optional calibration, resampling, band-pass (high/low-pass), non-wear, and sleep detection.

    Parameters
        :param data: (str | os.PathLike | Path | pandas.DataFrame) Path to the raw device file
            (e.g., Axivity .cwa, GENEActiv .bin, ActiGraph .gt3x, or supported CSV), or preloaded dataframe.
        :param subject_id: (str, optional) Identifier used in downstream metadata; if omitted, the placeholder 'none'
            is used.
        :param resample_rate: (int | 'infer', default 'infer') Target sampling rate in Hz for uniform resampling in case
            of sample rate jitter. Use 'infer', to keep/derive the device-native nominal rate, otherwise use a pos. Int.
        :param highpass_hz: (float | None, default 0.8) High-pass cutoff in Hz. Set to None to disable high-pass filter.
        :param lowpass_hz: (float | None, default 20.0) Low-pass cutoff in Hz. Set to None to disable low-pass filter.
            If both filters are set, must satisfy 0 < highpass_hz < lowpass_hz.
        :param calibrate : (bool, default True) If True, run device-specific post-hoc calibration.
        :param detect_nonwear: (bool, default True) If True, run non-wear detection.
        :param detect_sleep: (bool, default True) If True, run automated sleep detection using HDCZA (van Hees, 2018),
            giving both sleep period time windows (sptw), and higher-confidence sleep bouts.

    Returns
        :return: (Tuple[pd.DataFrame, Dict]) A time-indexed DataFrame containing the processed actigraphy data. The
            columns are time,x,y,z and, if enabled, boolean 'wear', 'sptw' (sleep period time window) and 'sleep window'
             masks. And an info dict, containing information about the applied processing steps and the data itself-"""

    if resample_rate != "infer":
        if not isinstance(resample_rate, int) or resample_rate <= 0:
            raise ValueError("'resample_rate' must be a positive int or 'infer'.")
    if highpass_hz is not None and highpass_hz <= 0:
        raise ValueError("'highpass_hz' must be > 0 or None.")
    if lowpass_hz is not None and lowpass_hz <= 0:
        raise ValueError("'lowpass_hz' must be > 0 or None.")
    if highpass_hz is not None and lowpass_hz is not None and not (highpass_hz < lowpass_hz):
        raise ValueError("When both filters are set, require 0 < 'highpass_hz' < 'lowpass_hz'.")

    device = ActimeterFactory(data, patient_id=subject_id or 'none')
    df = device.process(
        resample_rate=resample_rate, lowpass_hz=lowpass_hz, highpass_hz=highpass_hz, calibrate=calibrate,
        detect_nonwear=detect_nonwear, detect_sleep=detect_sleep)
    df = filter_sptws(df)  # filter to sptws that correspond to nights
    return df, device.get_info()


def plot(df: pd.DataFrame, *, return_axes: bool = False) -> Union[Figure, Tuple[Figure, Axes]]:
    """ Plot raw or processed actigraphy data. Processed data will be plotted with the
    corresponding overlays; otherwise the plot falls back to a raw-only view.

    Parameters
        :param df: (pandas.DataFrame) Time-indexed dataframe, either from .load() or from .process().
        :param return_axes: (bool, default False) If True, also return the Matplotlib Axes used for plotting.

    Returns
        :return fig: (matplotlib.figure.Figure) The created figure.
        :return (fig, ax): (tuple) The figure and axes, only if `return_axes` is True."""

    data_is_processed = utils.df_is_processed_and_valid(df)

    # warn if plotting more than 8 days (can be slow/visually dense)
    DAYS_THRESHOLD = 8
    idx = df.index
    if not isinstance(idx, pd.DatetimeIndex):  # try to coerce if index looks datetime-like
        coerced = pd.to_datetime(idx, errors='coerce')
        if coerced.notna().all():
            idx = coerced
    if isinstance(idx, pd.DatetimeIndex) and len(idx) > 0:
        span = idx.max() - idx.min()
        if span > pd.Timedelta(days=DAYS_THRESHOLD):
            logger.warning(
                f"Plotting a long span ({span / pd.Timedelta(days=1):.1f} days) "
                f"— consider downsampling or plotting a subset for readability.")

    fig, ax = draw_actigraphy_data(df, raw_only=not data_is_processed)
    return (fig, ax) if return_axes else fig


def compute_sleep_motor_features(processed_df: pd.DataFrame, subject_id: Optional[str] = None) -> pd.DataFrame:
    assert utils.df_is_processed_and_valid(processed_df), \
        (f"Invalid input DataFrame: expected processed actigraphy data with columns {utils.PROCESSED_REQUIRED_COLS};"
         f" missing {tuple(utils.missing_processed_columns(processed_df))}."
         f" Run actitect.api.process(...) or the CLI FileProcessor before calculate_sleep_features().")

    # only in sptws that corresponds to a night of sleep
    selected_nights = select_night_sptws(processed_df)
    logger.info(f"found {selected_nights.shape[0]} full nights of sleep in data.")

    # find movement bouts in sleep
    move_segment_mask, move_segment_ids, move_stats = segment_nocturnal_movements(processed_df, selected_nights)

    return compute_per_night_sleep_features(
        processed_df, selected_nights, move_segment_mask, move_segment_ids, sample_id=subject_id)


def compute_daytime_motor_features(df: pd.DataFrame) -> None:
    raise NotImplementedError()
