""" Contains functions to calculate global features, i.e. features describing an ensemble of several movement bouts,
typically from a whole night. Inputs are a full night dataframe of the processed data, i.e. containing the move column.
Returns are dicts containing the feature name and value. """
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.signal import find_peaks
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import MinMaxScaler

from .hopkins import Hopkins

__all__ = ['cluster_features', 'n_moves_per_time', 'movement_durations']


def cluster_features(move_segments, kde_kwargs: dict, peak_kwargs: dict, n_samples: int, sleep_wake_tolerance: float,
                     debug: bool = False):
    """ Quantify the clustering or dispersion of movements within processed nighttime data.

    Parameters:

        :param move_segments: (pd.DataFrame) of shape (n_moves, start_time, end_time, length) where each row represents
            a specific movement episode.
        :param kde_kwargs: (dict)
            Keyword arguments to be passed to the KernelDensity estimator.
        :param peak_kwargs: (dict)
            Keyword arguments to be passed to the peak finder.
        :param n_samples: (int)
            How many samples to use to evaluate the KDE.
        :param sleep_wake_tolerance: (float)
            Tolerance level to exclude areas near the sleep/wake boundaries from cluster analysis,
             in percent of data range.
        :param debug: (bool), optional
            If True, return additional debug information. (default=False)

    Returns:
        :return: (dict)
            Dictionary containing computed cluster features.

    Cluster Features:
        - iei_mean_sec: Mean value of inter-event intervals (IEI) in seconds.
        - iei_std_sec: Standard deviation of IEI in seconds.
        - iei_var_sec: Variance of IEI in seconds.
        - iei_cv: Coefficient of variation of IEI.
        - iei_iqr_sec: Inter-quartile range of IEI in seconds.
        - kde_n_peaks: Number of identified peaks in the kernel density estimate (KDE).
        - kde_avg_prom: Average prominence of identified peaks in KDE.
        - kde_max_prom: Maximum prominence of identified peaks in KDE.
        - kde_min_prom: Minimum prominence of identified peaks in KDE.
        - hopkins_score: Hopkins statistic score indicating cluster tendency.
    """

    cluster_feats = {}

    # inter-event intervals, i.e. durations between movements:
    iei = (move_segments.start_time.shift(-1) - move_segments.end_time).dt.total_seconds().to_numpy()[:-1]
    cluster_feats.update({
        'iei_mean_sec': np.mean(iei),
        'iei_std_sec': np.std(iei),
        'iei_var_sec': np.var(iei),
        'iei_cv': np.std(iei) / np.mean(iei),
        'iei_iqr_sec': np.quantile(iei, .75) - np.quantile(iei, .25)
    })

    # movement-midpoints, i.e. the timestamps in the middle of each movement:
    mmp = move_segments.start_time + (move_segments.end_time - move_segments.start_time) / 2

    # scale the values to [0,1] range for numerical stability and normalization:
    mmp_scaled = MinMaxScaler().fit_transform(mmp.to_numpy().reshape(-1, 1))
    cluster_feats.update({'hopkins_score': Hopkins(mmp_scaled, len(mmp_scaled)).score})  # hopkins score

    # kernel-density estimate (kde) to find movement clusters:
    kde = KernelDensity(**kde_kwargs)
    kde.fit(mmp_scaled)
    density_x = np.linspace(0, 1, n_samples).reshape(-1, 1)
    log_dens_scaled = kde.score_samples(density_x)
    density_y = np.exp(log_dens_scaled)

    peaks, peak_props = find_peaks(density_y, **peak_kwargs)  # peaks indicate clusters

    # exclude peaks if they are too close to sleep on/offset:
    exclude_indices = int(sleep_wake_tolerance * len(density_y))
    valid_mask = (peaks > exclude_indices) & (peaks < len(density_y) - exclude_indices)
    valid_peaks = peaks[valid_mask]
    valid_prominences = peak_props['prominences'][valid_mask]

    cluster_feats.update({'kde_n_peaks': len(valid_peaks)})
    if len(valid_prominences) > 0:
        cluster_feats.update({
            'kde_avg_prom': np.mean(valid_prominences),
            'kde_max_prom': np.max(valid_prominences),
            'kde_min_prom': np.min(valid_prominences),
        })
    else:
        cluster_feats['kde_avg_prom'] = cluster_feats['kde_max_prom'] = cluster_feats['kde_min_prom'] = 0

    if debug:
        return cluster_feats, (kde, mmp_scaled, density_x.flatten(), density_y.flatten(), valid_peaks)

    else:
        return cluster_feats


def n_moves_per_time(move_segments: pd.DataFrame):
    """ Compute the number of movements per hour from processed night data.

    Parameters:
        :param move_segments: (pd.DataFrame) of shape (n_moves, start_time, end_time, length) where each row represents
            a specific movement episode.

    Returns:
        :return: (dict) Dictionary containing computed movement features.

    Movement Features:
        - n_moves_per_h: Number of movements per hour. """
    n_moves = len(move_segments)
    total_time_h = (move_segments.end_time.max() - move_segments.start_time.min()).total_seconds() / (60 * 60)
    return {'n_moves_per_h': n_moves / total_time_h}


def movement_durations(move_segments: pd.DataFrame):
    """ Compute mode properties of movement duration distribution from processed night data.

    Parameters:
        :param move_segments: (pd.DataFrame)
            DataFrame containing movement episodes, where each row represents a specific movement episode.

    Returns:
        :return: (dict) Dictionary containing computed statistics of movement durations.

    Movement Duration Features:
        - mean_dur_sec: Mean duration of movement episodes in seconds.
        - std_dur_sec: Standard deviation of movement durations in seconds.
        - skew_dur_sec: Skewness of movement duration distribution.
        - kurt_dur_sec: Kurtosis of movement duration distribution.
        - q25_dur_sec: 25th percentile (1st quartile) of movement durations in seconds.
        - med_dur_sec: Median duration of movement episodes in seconds.
        - q75_dur_sec: 75th percentile (3rd quartile) of movement durations in seconds.

    ---
    Notes:
        - The 'move_segments' DataFrame should contain columns for 'start_time', 'end_time',
            and 'length(unit)' (duration) of each movement episode. Unit=seconds is assumed. """
    duration_feats = {}
    _dur_col = move_segments.columns[-1]
    episode_durations = move_segments[_dur_col]
    duration_feats.update({
        'mean_dur_sec': np.mean(episode_durations),
        'std_dur_sec': np.std(episode_durations),
        'skew_dur_sec': scipy.stats.skew(episode_durations),
        'kurt_dur_sec': scipy.stats.kurtosis(episode_durations),
    })

    duration_feats.update({
        name: value for name, value in zip(
            [f"q25_dur_sec", f"med_dur_sec", f"q75_dur_sec"], np.quantile(episode_durations, (.25, .5, .75))
        )
    })

    return duration_feats

