""" Functions to generate descriptive motion features from time-series data."""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from .. import utils
from . import global_ as global_feats
from . import local_ as local_feats

logger = logging.getLogger(__name__)

__all__ = ['CalcLocalMoveFeatures', 'CalcGlobalMoveFeatures']


@dataclass
class FeatureSetup:
    LOG_TIME_INFO_THRESHOLD_SEC: int = 3
    SPECTRAL_TARGET_FREQS: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 16])
    SPECTRAL_N_DOM_FREQS: int = 3
    PEAKS_PROM_THRESHOLD: float = .25
    PEAKS_MIN_DISTANCE: float = .2
    AC_PEAK_PROM_THRESHOLD_AUTO_CORR: float = .1
    NOLD_EMB_DIM: int = 3
    CLUSTER_KDE_KWARGS: Dict[str, Any] = field(default_factory=lambda: {'kernel': 'gaussian', 'bandwidth': .04})
    CLUSTER_N_KDE_SAMPLES: int = 400
    CLUSTER_PEAK_KWARGS_KDE: Dict[str, Any] = field(init=False)
    CLUSTER_SLEEP_WAKE_TOLERANCE: float = .075

    def __post_init__(self):
        self.CLUSTER_PEAK_KWARGS_KDE = {
            'prominence': .15, 'distance': int(.1 * (self.CLUSTER_N_KDE_SAMPLES - 1)), 'height': .5, }


class CalcLocalMoveFeatures:
    """ Calculate local, i.e. per movement bout, features from the data."""

    def __init__(self, mode: str, sample_rate: float):
        self.mode = mode
        self.sample_rate = sample_rate
        self.setup = FeatureSetup()
        self.sample_rate = sample_rate
        assert self.mode in ('max_version', 'max_version_with_processing', 'per_night'), \
            f"'mode' must be in ('max_version', 'max_version_with_processing', 'per_night'), got {self.mode}"

    def get_feature_names(self):
        """Helper function to get a complete list of the feature name strings."""
        _feature_func = None
        if self.mode == 'per_night':
            _feature_func = self.per_night_version
        else:
            raise ValueError(f"invalid 'mode' option: '{self.mode}'.")
        return list(_feature_func(*[pd.Series(np.random.normal(0, 1, 100)) for _ in range(4)]).keys())

    def calc_features(self, _movement_episode_df):
        """ Entry point to call the """
        _feature_func = None
        if self.mode == 'per_night':
            _feature_func = self.per_night_version
        else:
            raise ValueError(f"invalid 'mode' option: '{self.mode}'.")
        return _feature_func(
            _movement_episode_df.mag, _movement_episode_df.x, _movement_episode_df.y, _movement_episode_df.z)

    @staticmethod
    def _run_feature_func(feature_func, feature_name: str, warn_threshold: float, *args, **kwargs):
        """ Runs the given feature function, measures execution time, and returns a dict of features. """
        with utils.Timer() as timer:
            features = feature_func(*args, **kwargs)
        elapsed = timer()
        if elapsed > warn_threshold:
            size_str = f"({args[0].shape[0]},)" if args else ""
            logger.warning(f"{feature_name}: {size_str} -> {elapsed:.2f}s")
        return features

    def per_night_version(self, acc_mag, acc_x, acc_y, acc_z):
        for prop in (acc_mag, acc_x, acc_y, acc_z):
            assert isinstance(prop, pd.Series) or isinstance(prop, np.ndarray), \
                f"input must be of type pd.Series or np.array, not {type(prop)}"

        feature_calls = {
            'moments': dict(func=local_feats.moment_features, args=(acc_mag, acc_x, acc_y, acc_z), kwargs={}),
            'quantiles': dict(func=local_feats.quantile_features, args=(acc_mag, acc_x, acc_y, acc_z), kwargs={}),
            'energy': dict(func=local_feats.energy_features, args=(acc_mag, acc_x, acc_y, acc_z),
                           kwargs={}),
            'spectral': dict(func=local_feats.spectral_features, args=(acc_mag,), kwargs=dict(
                n_dom_freqs=self.setup.SPECTRAL_N_DOM_FREQS,
                target_frequencies_hz=self.setup.SPECTRAL_TARGET_FREQS, sample_rate=self.sample_rate)),
            'auto_corr': dict(func=local_feats.autocorr_features, args=(acc_mag, acc_x, acc_y, acc_z), kwargs=dict(
                sample_rate=self.sample_rate, peak_prom_threshold=self.setup.AC_PEAK_PROM_THRESHOLD_AUTO_CORR)),
            'peaks': dict(func=local_feats.peak_features, args=(acc_mag,), kwargs=dict(
                sample_rate=self.sample_rate, peak_prom_threshold=self.setup.PEAKS_PROM_THRESHOLD,
                min_peak_distance=self.setup.PEAKS_MIN_DISTANCE)),
            'nold': dict(func=local_feats.non_linear_dynamic_features, args=(acc_mag, acc_x, acc_y, acc_z),
                         kwargs=dict(emb_dim=self.setup.NOLD_EMB_DIM)),
            'poincare': dict(func=local_feats.poincare_features, args=(acc_mag,), kwargs={})}

        local_features = {}
        for feature_name, spec in feature_calls.items():
            local_features.update(self._run_feature_func(
                spec['func'], feature_name, self.setup.LOG_TIME_INFO_THRESHOLD_SEC,*spec['args'], **spec['kwargs']))
        return local_features


class CalcGlobalMoveFeatures:
    def __init__(self, global_move_segments_df: pd.DataFrame, mode: str, create_cluster_plots: bool):
        self.mode = mode
        self.create_cluster_plots = create_cluster_plots
        self.cluster_fig = None
        self.setup = FeatureSetup()
        if self.mode == 'per_night':
            self.features = self.per_night_version(global_move_segments_df)

        else:
            raise ValueError(f"The feature mode {self.mode} is unknown. Currently available are: 'per_night'")

    def per_night_version(self, move_segments: pd.DataFrame):
        global_features = {}

        # cluster features:
        _debug = True if self.create_cluster_plots else False
        cluster_feats, debug_info = global_feats.cluster_features(
            move_segments,
            kde_kwargs=self.setup.CLUSTER_KDE_KWARGS,
            peak_kwargs=self.setup.CLUSTER_PEAK_KWARGS_KDE,
            n_samples=self.setup.CLUSTER_N_KDE_SAMPLES,
            sleep_wake_tolerance=self.setup.CLUSTER_SLEEP_WAKE_TOLERANCE,
            debug=_debug
        )
        global_features.update(cluster_feats)

        if self.create_cluster_plots:
            kde, mmp_scaled, density_x, density_y, peaks = debug_info
            x_peaks, y_peaks = density_x[peaks], density_y[peaks]

            plt.rcParams['font.weight'] = 'bold'
            plt.rcParams['axes.labelweight'] = 'bold'
            fig, (lgd_ax, ax) = plt.subplots(2, 1, sharex=True, figsize=(9, 6), height_ratios=(.1, 1))
            plt.subplots_adjust(hspace=0)
            min_, max_ = np.max(density_y) / 40, np.max(density_y) / 5
            ax.scatter(mmp_scaled,
                       MinMaxScaler((min_, max_)).fit_transform(np.random.normal(0, 1, len(mmp_scaled)).reshape(-1, 1)),
                       c='c', edgecolor='k', marker='o', s=15, alpha=.5, label=f'move episodes (midpoint)')
            ax.plot(density_x, density_y, c='k', label=f"kde: {self.setup.CLUSTER_KDE_KWARGS['kernel']}"
                                                       f"(bw={self.setup.CLUSTER_KDE_KWARGS['bandwidth']})")
            # ax.scatter(x_peaks, y_peaks, marker='*', c='r', s=50, zorder=200)
            ax.axvspan(0, self.setup.CLUSTER_SLEEP_WAKE_TOLERANCE, color='r', alpha=.2, zorder=0)
            ax.axvspan(1 - self.setup.CLUSTER_SLEEP_WAKE_TOLERANCE, 1, color='r', alpha=.2, zorder=0,
                       label='excluded sleep on/offset')
            ax.plot(-1, 0, c='b', lw=2, label=f"n_peaks={cluster_feats['kde_n_peaks']}"
                                              f"(prom. > {self.setup.CLUSTER_PEAK_KWARGS_KDE['prominence']})")
            for _xp in x_peaks:
                ax.axvline(_xp, c='b', lw=.7, alpha=.5)
            ax.fill_between(density_x, density_y, color='gray', alpha=.2)
            for key in ('kde_avg_prom', 'kde_min_prom', 'kde_max_prom', 'hopkins_score'):
                ax.plot(0, 0, visible=False, label=f"{key}: {cluster_feats[key]:.3f}")
            ax.set_ylim(0)
            ax.set_xlim(0, 1)
            handles, labels = ax.get_legend_handles_labels()
            lgd_ax.legend(handles=handles, labels=labels, ncols=3, loc='lower left', bbox_to_anchor=(0, 0), fontsize=9)
            lgd_ax.axis('off')
            ax.set_xlabel('normalized sleep-period', labelpad=15, fontsize=13)
            ax.set_ylabel('probability density', labelpad=15, fontsize=13)
            ax.tick_params(which='major', tickdir='in', size=6, width=1.5, labelsize=11, right=True, top=True)
            ax.minorticks_on()
            ax.tick_params(which='minor', tickdir='in', size=3, width=.6, right=True, top=True)
            self.cluster_fig = fig
            plt.close()

        # moves per time:
        global_features.update(global_feats.n_moves_per_time(move_segments))

        # move durations:
        global_features.update(global_feats.movement_durations(move_segments))

        return global_features
