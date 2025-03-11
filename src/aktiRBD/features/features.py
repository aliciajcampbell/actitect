"""
Functions to generate the features from the raw time-series.
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from sklearn.preprocessing import MinMaxScaler

import aktiRBD.features.global_ as global_functions
import aktiRBD.features.local_ as local_functions
import aktiRBD.utils as utils

logger = logging.getLogger(__name__)

__all__ = ['CalcLocalMoveFeatures', 'CalcGlobalMoveFeatures']

# TODO: standardize time-series before feature extraction?? if so, all axis independent?
#  (if so, only on training data! -> on per patient/ per night basis or whole train dataset)
# store features per night or make separation afterwards?? (i have this function in datagenerator right)
# -> simply make night indicator in dataframe! (base on aggregate sptws during night window!)

# IDEAS: - spectral: e.g. turning over should have different frequencies then jerk's - time of movement also
# important - temporal information: order ot time data is important and not considered in max features? how to? (e.g.
# movement starts string and gets weaker and vice versa) (movement is relatively constant vs is spontaneous?)

# xy vs z have certain meanings, make use of that?
# I guess it's hard to make use of that since even though the axis are fixed with respect to wrist, wrist has
# too many degrees of freedom and orientation changes the whole time, i.e. there is no distinct orientation

#  standardize features in 0,1 range??? -> apparently not so critical for trees like xgboost

# include xy/z in moments/quantiles..? -> just include and check correlation...

# test v = median_filter(v, size=5, mode='nearest') to smoothen signal??

"""
TODO features:

stepcount: 
- moment features: mean, avg, ske, curt 
    > (of mag seems enough, see correlation plots)

    
- quantile features: min(0), 25, median(50), 75, max(100)
    > how data is distributed within movement (25 means 25% of data is below this values)
    
- spectral features: spectral entropy, psd per f bin, average power, dominant freqs + (psds), 
    > entropy of freq spectrum: higher: more complex/spread-out, lower: more concentrated, regular

- auto-correlation features:
    > cross-correlation but with time-shifted version of signal itself 
    > estimates periodic patterns, e.g. max indicates how periodic, loc of max indicates the periodicity,...
    
- Peak features:
    > e.g. total number of peaks 
    > prominence of peaks (min/max/avg) -> how pronounced or standing out the peaks are -> how jerky movement??
    
    
my ideas: 
- sample entropy: 
    > regularity/predictability of time series
    > degree of self-similarity between patterns within time-series by comparing similarity of subsequences of 
      fixed length m by calc. log likelihood that two seq. of length m+1 remain similiar within tolerance
    > lower values indicate higher regularitiy or preditability, higher means more irregular or 
      unpredictable patterns (SOUNDS GOOD)
    > used for ECG/EEG to assess the complexity and pred. of physiological processes
    
- clusterdness:
    > how to measure? by evaluating movement distance (like paper)
    
- derivatives: eg. derivative of acceleration dt3 is "jerk" (https://en.wikipedia.org/wiki/Jerk_(physics))
    > higher orders: dt4 = snap(jounce), dt5 = crackle, dt6 = pop 

- use sleep efficiency per night (from sleep analysis, i guess it's simply sleep bout/sptw)
    > might be hard to justify the quality of sleep-bouts / sptws
    
      
paper:
- duration of movements:
    > classify them as short/medium/long and get percentages? or simply mean etc. 
- total number of movements
- activity (i guess that was movements per sliding window? read again)
- activityrate (low/med/high percentages)
- movement strengt (low/med/high)
-> rather then using hard coded values something like quartiles might be better! (generalization)

    
    
from max:
- velocity not too bad -> but might also be just redundant information...

"""

"""
general thoughts:

- would it make sense to concatenate all movements and calc. features based on that? 
    > probably not, loss of information


- todo: make psd plot of night's: per night, patient, HC vs RBD, ...
- maybe adjust lower frequency bound again (higher)
"""


class CalcLocalMoveFeatures:
    """ Calculate local, i.e. per movement bout, features from the data."""

    def __init__(self, mode: str):
        self.mode = mode
        assert self.mode in ('max_version', 'max_version_with_processing', 'per_night'), \
            f"'mode' must be in ('max_version', 'max_version_with_processing', 'per_night'), got {self.mode}"

    def get_feature_names(self):
        """Helper function to get a complete list of the feature name strings."""
        _feature_func = None
        if self.mode in ('max_version', 'max_version_with_processing'):
            _feature_func = self.max_version
        elif self.mode == 'per_night':
            _feature_func = self.per_night_version
        return list(_feature_func(*[pd.Series(np.random.normal(0, 1, 100)) for _ in range(4)]).keys())

    def calc_features(self, _movement_episode_df):
        """ Entry point to call the """
        _feature_func = None
        if self.mode in ('max_version', 'max_version_with_processing'):
            _feature_func = self.max_version
        elif self.mode == 'per_night':
            _feature_func = self.per_night_version
        return _feature_func(
                _movement_episode_df.mag,
                _movement_episode_df.x,
                _movement_episode_df.y,
                _movement_episode_df.z
            )

    @staticmethod
    def per_night_version(acc_mag, acc_x, acc_y, acc_z):
        for prop in (acc_mag, acc_x, acc_y, acc_z):
            assert isinstance(prop, pd.Series) or isinstance(prop, np.ndarray), \
                f" input must be of type pd.Series or np.array, not {type(prop)}"

        local_features = {}
        LOG_TIME_INFO_THRESHOLD_SEC = 3

        # moment features
        with utils.Timer() as moment_timer:
            local_features.update(local_functions.moment_features(acc_mag, acc_x, acc_y, acc_z))
        if moment_timer() > LOG_TIME_INFO_THRESHOLD_SEC:
            logger.warning(f"moments: ({acc_mag.shape[0]},) -> {moment_timer()}s)")

        # quantile features:
        with utils.Timer() as quantile_timer:
            local_features.update(local_functions.quantile_features(acc_mag, acc_x, acc_y, acc_z))
        if quantile_timer() > LOG_TIME_INFO_THRESHOLD_SEC:
            logger.warning(f"quantiles: ({acc_mag.shape[0]},) -> {quantile_timer()}s)")

        # energy features:
        with utils.Timer() as energy_timer:
            local_features.update(local_functions.energy_features(acc_mag, acc_x, acc_y, acc_z))
        if energy_timer() > LOG_TIME_INFO_THRESHOLD_SEC:
            logger.warning(f"energy: ({acc_mag.shape[0]},) -> {energy_timer()}s)")

        # spectral features:
        with utils.Timer() as spectral_timer:
            # TODO: put in global config file
            TARGET_FREQS = [1, 2, 4, 8, 16]
            N_DOM_FREQS = 3
            SAMPLE_RATE = 100

            local_features.update(
                local_functions.spectral_features(
                    acc_mag, n_dom_freqs=N_DOM_FREQS,
                    target_frequencies_hz=TARGET_FREQS,
                    sample_rate=SAMPLE_RATE
                )
            )
        if spectral_timer() > LOG_TIME_INFO_THRESHOLD_SEC:
            logger.warning(f"spectral: ({acc_mag.shape[0]},) -> {spectral_timer()}s)")

        # auto-corr features:
        with utils.Timer() as auto_corr_timer:
            PEAK_PROM_THRESHOLD_AUTO_CORR = .1  # TODO: put in global config file

            local_features.update(
                local_functions.autocorr_features(
                    acc_mag, acc_x, acc_y, acc_z,
                    sample_rate=SAMPLE_RATE,
                    peak_prom_threshold=PEAK_PROM_THRESHOLD_AUTO_CORR,
                )
            )
        if auto_corr_timer() > LOG_TIME_INFO_THRESHOLD_SEC:
            logger.warning(f"auto_corr: ({acc_mag.shape[0]},) -> {auto_corr_timer()}s)")

        # peak features:
        with utils.Timer() as peak_timer:
            PEAK_PROM_THRESHOLD_PEAK_FEATS = .25
            MIN_PEAK_DISTANCE_PEAK_FEATS = .2

            local_features.update(
                local_functions.peak_features(
                    acc_mag,
                    sample_rate=SAMPLE_RATE,
                    peak_prom_threshold=PEAK_PROM_THRESHOLD_PEAK_FEATS,
                    min_peak_distance=MIN_PEAK_DISTANCE_PEAK_FEATS
                )
            )
        if peak_timer() > LOG_TIME_INFO_THRESHOLD_SEC:
            logger.warning(f"peaks: ({acc_mag.shape[0]},) -> {peak_timer()}s")

        # non-linear dynamic features:
        with utils.Timer() as nold_timer:
            EMB_DIM = 3  # TODO: put in global config file
            local_features.update(
                local_functions.non_linear_dynamic_features(
                    acc_mag, acc_x, acc_y, acc_z,
                    emb_dim=EMB_DIM
                )
            )
        if nold_timer() > LOG_TIME_INFO_THRESHOLD_SEC:
            logger.warning(f"nold: ({acc_mag.shape[0]},) -> {nold_timer()}s")

        # poincaré features:
        with utils.Timer() as poincare_timer:
            local_features.update(local_functions.poincare_features(acc_mag))
        if poincare_timer() > LOG_TIME_INFO_THRESHOLD_SEC:
            logger.warning(f"poincare: ({acc_mag.shape[0]},) -> {poincare_timer()}s")

        return local_features

    @staticmethod
    def max_version(acc_mag, acc_x, acc_y, acc_z):
        features = {}
        temp = {}

        # fastness of standing up    (what does that have to do with standing up?)
        features['mean_acc'] = np.mean(acc_mag)
        features['std_acc'] = np.mean(acc_mag)  # this seems to be a typo? should be np.std?

        # sum of absolute values:
        features['sma'] = np.sum(np.abs(acc_x)) + np.sum(np.abs(acc_y)) + np.sum(np.abs(acc_z))

        #  magnitude (isn't that just the same as 'acc'?) -> no, here it's the sum over whole series
        features['acc_pow'] = np.linalg.norm([acc_x, acc_y, acc_z])

        for axis in ['x', 'y', 'z']:
            data = eval(f'acc_{axis}')
            temp[f'velocity_{axis}'] = np.cumsum(data)
            features[f'mean_acc_{axis}'] = np.mean(data)
            features[f'std_acc_{axis}'] = np.std(data)
            features[f'max_acc_{axis}'] = np.max(data)
            features[f'min_acc_{axis}'] = np.min(data)
            features[f'median_acc_{axis}'] = np.median(data)
            features[f'rmse_acc_{axis}'] = np.sqrt(np.mean(data ** 2))
            features[f'skew_acc_{axis}'] = scipy.stats.skew(data)
            features[f'kurtosis_acc_{axis}'] = scipy.stats.kurtosis(data)

            features[f'start_vel_{axis}'] = np.mean(temp[f'velocity_{axis}'][0:50])
            features[f'mean_vel_{axis}'] = np.mean(temp[f'velocity_{axis}'])
            features[f'std_vel_{axis}'] = np.std(temp[f'velocity_{axis}'])
            features[f'max_vel_{axis}'] = np.max(temp[f'velocity_{axis}'])
            features[f'min_vel_{axis}'] = np.min(temp[f'velocity_{axis}'])
            features[f'median_vel_{axis}'] = np.median(temp[f'velocity_{axis}'])
            features[f'rmse_vel_{axis}'] = np.sqrt(np.mean(temp[f'velocity_{axis}'] ** 2))
            features[f'skew_vel_{axis}'] = scipy.stats.skew(temp[f'velocity_{axis}'])
            features[f'kurtosis_vel_{axis}'] = scipy.stats.kurtosis(temp[f'velocity_{axis}'])

            # temp[f'power_spectrum_{axis}'] = np.abs(np.fft.fft(accelerometer)) ** 2
            # features[f'energy_low_freq_{axis}'] = np.sum(temp[f'power_spectrum_{axis}'][0:6])
            # features[f'energy_high_freq_{axis}'] = np.sum(temp[f'power_spectrum_{axis}'][6:21])
            # features[f'auto_corr_{axis}'] = np.mean(np.correlate(accelerometer, accelerometer, mode='full'))

        return features


class CalcGlobalMoveFeatures:
    def __init__(self, global_move_segments_df: pd.DataFrame, mode: str, create_cluster_plots: bool):
        self.mode = mode
        self.create_cluster_plots = create_cluster_plots
        self.cluster_fig = None

        if self.mode == 'per_night':
            self.features = self.per_night_version(global_move_segments_df)

        else:
            raise ValueError(f"The feature mode {self.mode} is unknown. Currently available are: 'per_night'")

    def per_night_version(self, move_segments: pd.DataFrame):
        global_features = {}

        # cluster features:

        # TODO: put in global config gile
        KDE_KWARGS = {'kernel': 'gaussian', 'bandwidth': .04}
        N_KDE_SAMPLES = 400
        PEAK_KWARGS_KDE = {'prominence': .15, 'distance': int(.1 * (N_KDE_SAMPLES - 1)), 'height': .5}
        SLEEP_WAKE_TOLERANCE = .075
        DEBUG = True if self.create_cluster_plots else False

        cluster_feats, debug_info = global_functions.cluster_features(
                move_segments,
                kde_kwargs=KDE_KWARGS,
                peak_kwargs=PEAK_KWARGS_KDE,
                n_samples=N_KDE_SAMPLES,
                sleep_wake_tolerance=SLEEP_WAKE_TOLERANCE,
                debug=DEBUG
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
            ax.scatter(
                mmp_scaled,
                MinMaxScaler((min_, max_)).fit_transform(np.random.normal(0, 1, len(mmp_scaled)).reshape(-1, 1)),
                c='c', edgecolor='k',
                marker='o', s=15, alpha=.5,
                label=f'move episodes (midpoint)'
            )

            ax.plot(density_x, density_y, c='k', label=f"kde: {KDE_KWARGS['kernel']}(bw={KDE_KWARGS['bandwidth']})")
            # ax.scatter(x_peaks, y_peaks, marker='*', c='r', s=50, zorder=200)
            ax.axvspan(0, SLEEP_WAKE_TOLERANCE, color='r', alpha=.2, zorder=0)
            ax.axvspan(1 - SLEEP_WAKE_TOLERANCE, 1, color='r', alpha=.2, zorder=0, label='excluded sleep on/offset')

            ax.plot(
                -1, 0,
                c='b', lw=2,
                label=f"n_peaks={cluster_feats['kde_n_peaks']} (prom. > {PEAK_KWARGS_KDE['prominence']})"
            )
            for _xp in x_peaks:
                ax.axvline(_xp, c='b', lw=.7, alpha=.5)
            ax.fill_between(density_x, density_y, color='gray', alpha=.2)

            for key in ('kde_avg_prom', 'kde_min_prom', 'kde_max_prom', 'hopkins_score'):
                ax.plot(0, 0, visible=False, label=f"{key}: {cluster_feats[key]:.3f}")
            ax.set_ylim(0)
            ax.set_xlim(0, 1)
            handles, labels = ax.get_legend_handles_labels()
            lgd_ax.legend(
                handles=handles, labels=labels,
                ncols=3,
                loc='lower left', bbox_to_anchor=(0, 0),
                fontsize=9
            )
            lgd_ax.axis('off')

            ax.set_xlabel('normalized sleep-period', labelpad=15, fontsize=13)
            ax.set_ylabel('probability density', labelpad=15, fontsize=13)
            ax.tick_params(which='major', tickdir='in', size=6, width=1.5, labelsize=11, right=True, top=True)
            ax.minorticks_on()
            ax.tick_params(which='minor', tickdir='in', size=3, width=.6, right=True, top=True)
            self.cluster_fig = fig
            plt.close()

        # moves per time:
        global_features.update(global_functions.n_moves_per_time(move_segments))

        # move durations:
        global_features.update(global_functions.movement_durations(move_segments))

        return global_features
