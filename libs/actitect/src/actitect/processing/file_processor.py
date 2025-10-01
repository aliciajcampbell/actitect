import gc
import json
import logging
from pathlib import Path
from argparse import Namespace

from typing import Optional, TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow
import pyarrow.parquet as parquet

from .movements import segment_nocturnal_movements
from .. import utils
from ..features import CalcLocalMoveFeatures, CalcGlobalMoveFeatures
from ..vis import draw_actigraphy_data
if TYPE_CHECKING:
    from ..actimeter.basedevice import ResolveNwSleepParams

MIN_LOCAL_SAMPLE_LENGTH_SECONDS = 0.5
MAX_LOCAL_SAMPLE_LENGTH_SECONDS = 50

logger = logging.getLogger(__name__)

__all__ = ['FileProcessor']


class FileProcessor:
    _delete_confirmation_given = False  # only relevant for 'delete_processed_files' operational modes

    def __init__(self, patient_id: str, record_id: str, label: str, acti_file_path: Path, save_dir: Path,
                 save_processed_data: bool, sleep_log: pd.DataFrame = None, ax6_legacy_mode: bool = False):

        self.patient_id = patient_id
        self.record_id = record_id if record_id and record_id != 'none' else None
        self.saving_suffix = self.patient_id if not self.record_id else f"{self.patient_id}_{self.record_id}"
        self.label = label
        self.acti_file_path = acti_file_path
        self.save_processed_data = save_processed_data
        self.sleep_log = sleep_log
        self.ax6_legacy_mode = ax6_legacy_mode

        self.recording_save_dir = save_dir.joinpath(patient_id) if not self.record_id \
            else save_dir.joinpath(patient_id, self.record_id)
        self.parquet_path = self.recording_save_dir.joinpath(f"df-{self.saving_suffix}.parquet")
        self.info, self.process_kwargs, self.feat_kwargs, self.pbar = None, None, None, None

    def process(self, feat_kwargs: dict, process_kwargs: dict, operational_kwargs: Namespace, pbar=None):
        """ Process the raw data and calculate features based on the specified mode.
        Parameters:
            :param feat_kwargs: (dict)
                Dictionary containing parameters related to the feature calculation process.
            :param process_kwargs: (dict)
                Dictionary containing parameters related to the data processing steps, such as filtering and resampling.
            :param pbar: (tqdm object, optional)
                Progress bar instance for displaying progress. Default is None.
            :param operational_kwargs: (argparse.Namespace) The command line arguments defining the operational mode of
                the processing function. Refer to  main README.md or process_dataset.py for details. """

        self.pbar = pbar
        self.feat_kwargs = feat_kwargs
        self.process_kwargs = process_kwargs

        if self.pbar:
            self.pbar.set_description(f"[PROGRESS]: Processing files")
            self.pbar.set_postfix({
                "file": f"{self.saving_suffix}",
                # "save_processed": operational_kwargs.save_processed,
                # "create_plots": operational_kwargs.create_plots,
                # "redo_processing": operational_kwargs.redo_processing,
                # "skip_feature_calc": operational_kwargs.skip_feature_calc,
                # "delete_processed_files": operational_kwargs.delete_processed_files
            })

        # 0. handle the deletion case (cleanup mode)
        if operational_kwargs.delete_processed_files:
            self._validate_and_perform_deletion()
            return  # exit process method

        # 1. apply processing if needed
        processed_df, self.info = self._load_or_process_data(operational_kwargs.redo_processing)
        assert self.info['processing']['all_steps_successful'] is True, \
            f"Incomplete pre-processing detected, excluding {self.saving_suffix} ..."

        # 2. feature calculation (if specified)
        if not operational_kwargs.skip_feature_calc:
            if self.pbar:
                self.pbar.set_postfix({"file": f"{self.saving_suffix}", "status": "extr. movements"})
            selected_nights, move_bout_mask, move_bout_ids = self._segment_nocturnal_movements(processed_df)
            local_feat_df, global_feat_df = self._calculate_features(
                processed_df, selected_nights, move_bout_mask, move_bout_ids)

            _feat_dir = utils.check_make_dir(self.recording_save_dir.joinpath(f"features/{self.feat_kwargs['mode']}/"))
            global_feat_df.to_csv(_feat_dir.joinpath(f"global-features-{self.saving_suffix}.csv"))
            local_feat_df.to_csv(_feat_dir.joinpath(f"local-features-{self.saving_suffix}.csv"))
            utils.dump_to_json(self.info, self.recording_save_dir.joinpath(f"info-{self.saving_suffix}.json"))
            logger.info(f"(io: {self.saving_suffix}): features successfully saved")

            # cleanup memory
            del local_feat_df, global_feat_df, selected_nights
        else:
            logger.info(f"(io: {self.saving_suffix}): Skipping feature extraction."
                        f"('args.skip_feature_calc'={operational_kwargs.skip_feature_calc})")

        # 3. plotting (if specified)
        if operational_kwargs.create_plots:
            self._plot_data(processed_df)

        del processed_df
        gc.collect()

    def _validate_and_perform_deletion(self):
        """ Delete the processed .parquet file if all derived files exist. prompts for confirmation once."""
        if not FileProcessor._delete_confirmation_given:  # prompt for confirmation on first class instance
            logger.warning("You are about to delete processed .parquet files. This action cannot be undone.")
            confirm = input("Continue? [y/n]: ")
            if confirm.lower() != 'y':
                raise SystemExit("Deletion operation cancelled by the user, terminating...")
            # set global class variable True for all future class instances in this runtime:
            FileProcessor._delete_confirmation_given = True

        if not self.has_been_processed:  # check if files exists in the first place
            logger.warning(f"Processed file for patient {self.saving_suffix} does not exist. Skipping deletion.")
            return  # Nothing to delete

        # define and scan for derived files that should exist(feature files and plots)
        feature_dir = self.recording_save_dir.joinpath("features")
        raw_plot_path = self.parquet_path.parent.joinpath(f"{self.saving_suffix}-raw.png")
        processed_plot_path = self.parquet_path.parent.joinpath(f"{self.saving_suffix}-processed.png")
        # search for feature files within all subdirectories of 'features/'
        global_feature_files = list(feature_dir.rglob(f"global-features-{self.saving_suffix}.csv"))
        local_feature_files = list(feature_dir.rglob(f"local-features-{self.saving_suffix}.csv"))
        required_files = global_feature_files + local_feature_files + [raw_plot_path, processed_plot_path]

        missing_files = [str(file) for file in required_files if not file.exists()]
        if not missing_files:
            try:
                self.parquet_path.unlink()
                logger.warning(f"({self.saving_suffix}) deleted processed data .parquet file {self.parquet_path}.")
            except Exception as e:
                logger.error(f"({self.saving_suffix}) failed to delete {self.parquet_path}: {e}")
        else:
            logger.warning(f"cannot delete {self.parquet_path} as the following derived files are missing:"
                           f"{missing_files}. Skipping deletion.")

    @property
    def has_been_processed(self):
        return self.parquet_path.exists()  # simply checks if the processed file exists

    def _load_info(self):
        with open(self.recording_save_dir.joinpath(f"info-{self.saving_suffix}.json"), 'r') as f:
            return json.load(f)

    def _load_or_process_data(self, redo_processing: bool):
        from ..actimeter import ActimeterFactory
        if self.has_been_processed and not redo_processing:  # file exist and we do not want to force re-processing
            processed_df = parquet.read_table(self.parquet_path).to_pandas()
            with open(self.recording_save_dir.joinpath(f"info-{self.saving_suffix}.json"), 'r') as f:
                info = json.load(f)
            logger.info(f"(io: {self.saving_suffix}): using archived pre-processed .parquet file.")

        else:  # otherwise, run processing from raw data
            if self.has_been_processed and redo_processing:
                logger.warning(f"(io: {self.saving_suffix}): processed data exists as .parquet file but will"
                               f"re-run processing due to operational args. Data will be overwritten "
                               f"if 'save_processed_data' is True ({self.save_processed_data}).")
            if self.ax6_legacy_mode:  # factory handles kwargs but this will mute the warning for other devices
                actimeter = ActimeterFactory(self.acti_file_path, self.saving_suffix, legacy_mode=self.ax6_legacy_mode)
            else:
                actimeter = ActimeterFactory(self.acti_file_path, self.saving_suffix)
            utils.check_make_dir(self.recording_save_dir, use_existing=True)
            processed_df = actimeter.process(**self.process_kwargs)

            info = {'meta': actimeter.meta, 'header': actimeter.binary_header, 'processing': actimeter.processing_info}
            if self.save_processed_data:
                parquet.write_table(pyarrow.Table.from_pandas(processed_df), self.parquet_path, compression='snappy')
                utils.dump_to_json(info, self.recording_save_dir.joinpath(f"info-{self.saving_suffix}.json"))
                logger.info(f"(io: {self.saving_suffix}): pre-processed DataFrame saved to .parquet file")
            del actimeter

        return processed_df, info

    def _segment_nocturnal_movements(self, processed_df: pd.DataFrame, params: Optional['ResolveNwSleepParams'] = None):
        from ..actimeter.basedevice import ResolveNwSleepParams

        params = params if params else ResolveNwSleepParams()

        # get non-wear segments: (start, end, length)
        non_wears = utils.extract_segments(
            processed_df, column='wear', condition=False, add_during_night_indicator=True,
            night_start=params.night_start, night_end=params.night_end, during_night_h_thres=.1)

        # get sptw segments: (start, end, length)
        sptws = utils.extract_segments(
            processed_df, column='sptw', condition=True, add_during_night_indicator=True,
            night_start=params.night_start, night_end=params.night_end,
            during_night_h_thres=params.night_sptw_duration_during_night_h)

        # select only sptws that correspond to nights: (i.e. begin/end in typical night window and long enough)
        sptws['selected'] = sptws.apply(
            lambda row: True if (row['during_night'] and row['length(h)'] > params.night_sptw_threshold_h) else False,
            axis=1)
        selected_sptws = sptws[sptws.selected].reset_index().rename(columns={'index': 'sptw_idx'})
        selected_sptws.index.name = 'night'
        logger.info(f"({self.saving_suffix}) found {selected_sptws.shape[0]} full nights of sleep in data.")

        # mask the movement bouts:
        move_segment_mask, move_segment_ids, move_stats = segment_nocturnal_movements(processed_df, selected_sptws)

        # update some infos about selected nights and number of movements:
        logger.info(
            f"({self.saving_suffix}) movement segmentation done: n_moves={len(move_segment_ids)} across all nights.")

        self.info['processing']['sleep_movements'].update({f"n_moves_total": len(move_segment_ids)})
        self.info['processing']['sleep_movements'].update(move_stats)
        selected_sptw_dict = selected_sptws.assign(**{'length(h)': lambda x: x['length(h)'].round(4)}).to_dict()
        self.info['processing']['sleep_segmentation'].update({'selected_sptw_nights': {
            index: {key: value_list[index] for key, value_list in selected_sptw_dict.items()}
            for index in range(len(next(iter(selected_sptw_dict.values()))))}})
        self.info['processing']['non-wear'].update({'final_segments': non_wears.to_dict()})

        del sptws
        del non_wears
        gc.collect()
        return selected_sptws, move_segment_mask, move_segment_ids

    def _calculate_features(self, _processed_df: pd.DataFrame, _selected_nights: pd.DataFrame,
                            _move_bout_mask: pd.DataFrame, _move_bout_ids: pd.DataFrame):

        sample_rate = self._get_final_sample_rate()
        local_feat_df, global_feat_df = pd.DataFrame(), pd.DataFrame()
        for night_idx, night_sptw in _selected_nights.iterrows():  # loop over nights
            if self.pbar:
                self.pbar.set_postfix(
                    {"file": f"{self.saving_suffix}", "night": f"{night_idx + 1}/{_selected_nights.shape[0]}"})

            # select the subset of the data corresponding to the night
            night_df = _processed_df[(_processed_df.index >= night_sptw['start_time'])
                                     & (_processed_df.index <= night_sptw['end_time'])]
            move_bout_mask_night = _move_bout_mask[(_move_bout_mask.index >= night_sptw['start_time'])
                                                   & (_move_bout_mask.index <= night_sptw['end_time'])]
            move_bout_ids_night = _move_bout_ids[(_move_bout_ids >= move_bout_mask_night.min())
                                                 & (_move_bout_ids <= move_bout_mask_night.max())]
            self.info['processing']['sleep_movements'].update({f"n_moves_night_{night_idx}": len(move_bout_ids_night)})

            if not len(move_bout_ids_night) > 10:
                logger.warning(f"not enough movement episodes found in {self.saving_suffix} night {night_idx}"
                               f"(n={len(move_bout_ids_night)}). Check for non-wear.")

            else:
                if self.feat_kwargs['mode'] == 'per_night':  # calc. global (per-night) features
                    global_feats = self._calc_global_features(night_df, night_sptw, night_idx)
                    global_feat_df = pd.concat([global_feat_df, pd.DataFrame(global_feats, index=[night_idx])])
                    del global_feats
                    gc.collect()

                # calculate local (per-move) features:
                feat_names = CalcLocalMoveFeatures(self.feat_kwargs['mode'], sample_rate).get_feature_names()
                for idx, relevant_idx in enumerate(move_bout_ids_night):  # loop over movement bouts

                    move_bout_df = night_df[move_bout_mask_night == relevant_idx]

                    if move_bout_df.empty:
                        logger.warning(f"move bouts not found in night_idx: {night_idx}, rel_idx: {relevant_idx}"
                                       f"\n - ids: {move_bout_ids_night}"
                                       f"\n - mask: {move_bout_mask_night}"
                                       f"\n - episode: {move_bout_df}")

                    local_feats = self._calc_local_features(
                        move_bout_df, relevant_idx, night_sptw, night_idx, feat_names, sample_rate)
                    local_feat_df = pd.concat([local_feat_df, pd.DataFrame(local_feats, index=[idx]).astype(object)])
                    if self.pbar:
                        self.pbar.set_postfix({"file": f"{self.saving_suffix}",
                                               "night": f"{night_idx + 1}/{_selected_nights.shape[0]} ",
                                               "features": f"{np.round((idx / len(move_bout_ids_night)) * 100, 1)}%"})
                    del local_feats
                    del move_bout_df
                    gc.collect()

            del night_df
            del move_bout_ids_night
            del move_bout_mask_night
            gc.collect()

        return local_feat_df, global_feat_df

    def _calc_global_features(self, _night_df: pd.DataFrame, _night_sptw: pd.DataFrame, _night_idx: int):

        # get a representation as start, end, length where each row corresponds to one movement
        night_move_segments = utils.extract_segments(_night_df, 'movement', True, time_unit='s')

        # apply a duration filter if specified:
        if isinstance(self.feat_kwargs['filter_move_durations'], tuple):
            _low, _high = self.feat_kwargs['filter_move_durations']
            logger.info(f"filtering movement durations: {_low}s <= t <= {_high}s")
            night_move_segments = night_move_segments[(night_move_segments['length(s)'] > _low)
                                                      & (night_move_segments['length(s)'] < _high)]

        with utils.Timer() as global_timer:  # calculate global features:
            calc_global_move_features = CalcGlobalMoveFeatures(night_move_segments, self.feat_kwargs['mode'],
                                                               self.feat_kwargs['create_cluster_plots'])
            _global_feats = calc_global_move_features.features

            if isinstance(calc_global_move_features.cluster_fig, plt.Figure):
                _cluster_dir = utils.check_make_dir(
                    self.recording_save_dir.joinpath(f"move_clusters/"), True, verbose=False)
                calc_global_move_features.cluster_fig.savefig(
                    _cluster_dir.joinpath(f"clusters-{self.saving_suffix}-night{_night_idx}.png"), bbox_inches='tight')

        sptw_duration = _night_sptw['length(h)']
        sleep_duration = (_night_df.index.to_series().diff()[_night_df['sleep_bout']] / pd.Timedelta(hours=1)).sum()
        _global_feats['sleep_eff'] = sleep_duration / sptw_duration
        _global_feats['waso'] = sptw_duration / sleep_duration
        _global_feats['id'] = self.patient_id
        _global_feats['record_id'] = self.record_id if self.record_id else 'none'
        _global_feats['diagnosis'] = self.label
        _global_feats['start_time'] = _night_sptw['start_time']
        _global_feats['end_time'] = _night_sptw['end_time']
        _global_feats['length(h)'] = _night_sptw['length(h)']
        _global_feats['sptw_idx'] = _night_sptw['sptw_idx']
        _global_feats['night'] = _night_idx
        _global_feats['runtime'] = global_timer()
        del night_move_segments

        return _global_feats

    def _calc_local_features(self, _move_bout_df: pd.DataFrame, _relevant_idx: int, _night_sptw: pd.DataFrame,
                             _night_idx: int, _feature_names: np.ndarray, sample_rate: float):

        with utils.Timer() as local_timer:
            if not (int(MIN_LOCAL_SAMPLE_LENGTH_SECONDS * sample_rate)
                    <= _move_bout_df.shape[0]
                    <= int(MAX_LOCAL_SAMPLE_LENGTH_SECONDS * sample_rate)):
                logger.debug(f"timeseries too short/long: {_relevant_idx} "
                             f"({(_move_bout_df.index[-1] - _move_bout_df.index[0]).total_seconds():.3f}s)")
                _local_feats = {_feat_name: np.NaN for _feat_name in _feature_names}
            else:  # calculate the movement features of each episode
                _feature_generator = CalcLocalMoveFeatures(self.feat_kwargs['mode'], sample_rate)
                _local_feats = _feature_generator.calc_features(_move_bout_df)

        _local_feats['id'] = self.patient_id
        _local_feats['record_id'] = self.record_id if self.record_id else 'none'
        _local_feats['diagnosis'] = self.label
        _local_feats['time_start'] = _move_bout_df.index[0]
        _local_feats['time_end'] = _move_bout_df.index[-1]
        _local_feats['time_diff'] = (_move_bout_df.index[-1] - _move_bout_df.index[0]).total_seconds()
        _local_feats['sptw_start'] = _move_bout_df.sptw.iloc[0]
        _local_feats['sptw_end'] = _move_bout_df.sptw.iloc[0]
        _local_feats['sptw_idx'] = _night_sptw['sptw_idx']
        _local_feats['night'] = _night_idx
        _local_feats['runtime'] = local_timer()
        return _local_feats

    def _get_final_sample_rate(self) -> float:
        """ Retrieves the final sample rate from either the device header or the processing info.
        Raises a ValueError if neither provides a valid numeric sample rate.
        If both are available and differ, a warning is logged.
        Assumes that self.info has already been populated.

        Returns:
            A numeric sample rate (float).
        """
        assert self.info is not None, \
            "self.info is not defined. Ensure data has been processed before calling this method."

        # try to access header sample rate if defined
        header_fs = self.info.get('header', {}).get('sample_rate')
        header_fs = self._validate_sample_rate_type(header_fs, self.saving_suffix) if header_fs else None

        # try to access the data sampling rate (either with or without resampling)
        data_fs = self.info['processing']['resampling'].get(
            'resample_fs_mean', self.info['processing']['resampling'].get('raw_fs_mean'))
        data_fs = self._validate_sample_rate_type(data_fs, self.saving_suffix) if data_fs else None

        # check if both are undefined
        if header_fs is None and data_fs is None:
            raise ValueError(f"(io: {self.saving_suffix}) No valid sample rate found in header or processing info.")
        # if one is undefined, use the defined one
        if header_fs is None:
            return data_fs
        elif data_fs is None:
            return header_fs
        else:  # both defined: warn if they differ and use data_fs
            if header_fs != data_fs:
                logger.warning(f"(io: {self.saving_suffix}) Device header sample rate ({header_fs} Hz) "
                               f"differs from data sample rate ({data_fs} Hz)."
                               f"Consider activating resampling in the processing pipeline.")
            return data_fs

    @staticmethod
    def _validate_sample_rate_type(sample_rate, log_suffix: str):
        if isinstance(sample_rate, str):
            try:
                sample_rate = float(sample_rate)
            except ValueError:
                logger.warning(f"(io: {log_suffix}) sample rate '{sample_rate}' cannot be converted to float.")
                return None
            return sample_rate
        elif isinstance(sample_rate, (float, int)):
            return sample_rate
        else:
            logger.warning(
                f"(io: {log_suffix}) sample rate '{sample_rate}' has invalid dtype: {type(sample_rate).__name__}")
            return None

    def _plot_data(self, processed_df: pd.DataFrame):
        """Plots raw and processed data, saving the plots to disk."""
        from ..actimeter import ActimeterFactory

        def _save_plot(fig, path, data_type):
            """Helper function to save a plot and log status."""
            if path.exists():
                logger.info(f"(io: {self.saving_suffix}): {data_type} plot exists and will be overwritten.")
            fig.savefig(path, bbox_inches='tight')
            plt.close(fig)
            del fig
            gc.collect()

        # Define paths
        raw_plot_path = self.parquet_path.parent.joinpath(f"{self.saving_suffix}-raw.png")
        processed_plot_path = self.parquet_path.parent.joinpath(f"{self.saving_suffix}-processed.png")

        # Plot raw data
        raw_df = ActimeterFactory(self.acti_file_path, self.saving_suffix).load_raw_data()
        fig_raw = draw_actigraphy_data(raw_df, self.sleep_log, raw_only=True)
        _save_plot(fig_raw, raw_plot_path, "raw")

        # Plot processed data
        fig_pr = draw_actigraphy_data(processed_df, self.sleep_log, raw_only=False)
        _save_plot(fig_pr, processed_plot_path, "processed")

        del raw_df
        gc.collect()
