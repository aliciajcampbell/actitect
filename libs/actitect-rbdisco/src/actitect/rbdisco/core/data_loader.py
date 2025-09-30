import logging
from collections import defaultdict
from pathlib import Path
from typing import Union, List

import numpy as np
import pandas as pd
from numba import jit
from sklearn.utils.class_weight import compute_class_weight

from actitect import utils
from actitect.classifier.tools.feature_set import FeatureSet
from actitect.config import RebalanceDatasetsConfig

__all__ = ['DataLoader']

logger = logging.getLogger(__name__)


class DataLoader:
    """Manages the retrieval of local and global feature files, processes metadata,
    applies feature aggregation strategies, and prepares train/test datasets."""

    def __init__(self, data_dir: Union[Path, List[Path]], meta_path: Union[Path, pd.DataFrame],
                 feature_dir: Path, aggregation: List[str],
                 included_local_features: List[str], included_global_features: List[str],
                 binary_mapping: dict, add_str_labels: bool = False, verbose: bool = True, shuffle: bool = True,
                 rebalance_datasets: RebalanceDatasetsConfig = None):

        self.dataset_is_pooled = isinstance(data_dir, list) and len(data_dir) > 1
        self.data_dirs = data_dir if isinstance(data_dir, list) else [data_dir]
        self.feature_dir = feature_dir  # relative to patient_dir
        self.local_feat_files = sorted([
            file for base_dir in self.data_dirs  # loop over patient dirs from datasets
            for file in list(base_dir.glob(f"*/{feature_dir}/*local*.csv"))  # single record/patient
                        + list(base_dir.glob(f"*/*/{feature_dir}/*local*.csv"))  # multi-record
        ])

        if len(self.local_feat_files) == 0:
            raise UserWarning(f"could not locate local feature files make sure '{self.data_dirs}' exists and "
                              f"contains patient subdirs populated with the feature files.")

        self.aggregation = aggregation
        self.binary_mapping = binary_mapping
        self.included_local_features = included_local_features
        self.included_global_features = included_global_features
        self.verbose = verbose
        self.add_str_labels = add_str_labels
        if isinstance(rebalance_datasets, dict):
            rebalance_datasets = RebalanceDatasetsConfig.from_dict(rebalance_datasets)
        self.rebalance_datasets = rebalance_datasets

        self.meta_df, self.records_train, self.records_test = self._get_meta(meta_path)

        if self.dataset_is_pooled:
            assert self.meta_df is not None and 'dataset_id' in self.meta_df.columns, \
                "When using multiple datasets, meta file must contain a 'dataset_id' column."
            assert not self.meta_df['dataset_id'].isnull().any(), \
                "'dataset_id' column in meta file contains missing values."
            duplicate_ids = self.meta_df.groupby('ID')['dataset_id'].nunique()
            conflicting_ids = duplicate_ids[duplicate_ids > 1].index.tolist()
            assert not conflicting_ids, f"Subject IDs found in multiple datasets: {conflicting_ids}"

            self.map_subject_to_dataset = self.meta_df.set_index('ID')['dataset_id'].to_dict()
            self.map_dataset_to_data_dir = self._infer_dataset_dir_mapping(self.meta_df, self.data_dirs)

        self.local_feat_files = self._check_and_filter_feature_files(self.local_feat_files, self.meta_df)
        n_hc_train, n_rbd_train, n_hc_test, n_rbd_test = self._get_class_counts(self.meta_df)
        if shuffle:
            np.random.shuffle(self.records_train)
            np.random.shuffle(self.records_test)

        self.train_id_map = self.test_id_map = None
        self.num_total_rbd = n_rbd_train + n_rbd_test
        self.num_total_hc = n_hc_train + n_hc_test
        self.num_total = self.num_total_rbd + self.num_total_hc
        if verbose:
            self._log_info_(n_rbd_train, n_hc_train, n_rbd_test, n_hc_test)

    def __str__(self):
        return f"DataLoader(n_rbd={self.num_total_rbd}, n_hc={self.num_total_hc})"

    def get_train_test_data(self, agg_level: str = 'night', agg_with_numba: bool = True):

        local_feat_df = self._create_local_feature_dataframe()
        x, y, y_str, feat_map, train_index_mask, test_index_mask = None, None, None, None, None, None

        if agg_level == 'move':
            _feature_df = local_feat_df[self.included_local_features]
            feat_map = _feature_df.columns.values
            x = _feature_df.to_numpy()
            y = local_feat_df.ground_truth
            y_str = local_feat_df.diagnosis if self.add_str_labels else None
            train_index_mask = local_feat_df.record_key.isin(self.records_train)
            test_index_mask = local_feat_df.record_key.isin(self.records_test)

            if self.dataset_is_pooled and getattr(self, 'rebalance_datasets', None) \
                    and getattr(self.rebalance_datasets, 'method', None) not in (None, 'none'):
                logger.warning(
                    "[Rebalance] requested (method=%s) but only implemented for agg_level='night'. Skipping.",
                    self.rebalance_datasets.method
                )

        else:

            feat_df_night = self._create_global_feature_dataframe(local_feat_df.copy(), use_numba=agg_with_numba)

            if agg_level == 'night':
                _included_local_features = [
                    f"{_feat}_{postfix}" for postfix in self.aggregation for _feat in self.included_local_features]
                _feature_df = feat_df_night[_included_local_features + self.included_global_features]

                feat_map = _feature_df.columns.values
                x = _feature_df.to_numpy()
                y = feat_df_night.ground_truth
                y_str = feat_df_night.diagnosis if self.add_str_labels else None
                train_index_mask = feat_df_night.record_key.isin(self.records_train)
                test_index_mask = feat_df_night.record_key.isin(self.records_test)
                self.train_id_map = feat_df_night[train_index_mask].id.to_numpy()
                self.test_id_map = feat_df_night[test_index_mask].id.to_numpy()

                # Apply dataset rebalancing (pooled only)
                if self.dataset_is_pooled and getattr(self, 'rebalance_datasets', None) \
                        and getattr(self.rebalance_datasets, 'method', None) not in (None, 'none'):
                    logger.warning(
                        f" applying composite dataset resampling with 'method='{self.rebalance_datasets.method}'")
                    train_index_mask = self._apply_dataset_rebalancing(feat_df_night, train_index_mask)
                    # refresh train_id_map after rebalancing
                    self.train_id_map = feat_df_night[train_index_mask].id.to_numpy()
                    self._log_rebalanced_train_summary(feat_df_night, train_index_mask)

            elif agg_level == 'patient':
                _included_local_features = [
                    f"{_feat}_{postfix}" for postfix in self.aggregation for _feat in self.included_local_features
                ]
                feat_df_patient = (
                    feat_df_night
                    .groupby(['id'])
                    [[*_included_local_features, *self.included_global_features, 'ground_truth']]
                    .agg('mean')  # numeric columns only, large entropy
                    .join(feat_df_night.groupby('id')[['record_key', 'record_id']].agg('first'))  # take first string/id
                    .reset_index()
                )
                _feature_df = feat_df_patient[_included_local_features + self.included_global_features]

                feat_map = _feature_df.columns.values
                x = _feature_df.to_numpy()
                y = feat_df_patient.ground_truth
                train_index_mask = feat_df_patient.record_key.isin(self.records_train)
                test_index_mask = feat_df_patient.record_key.isin(self.records_test)
                self.train_id_map = feat_df_patient[train_index_mask].id.to_numpy()
                self.test_id_map = feat_df_patient[test_index_mask].id.to_numpy()

                if self.dataset_is_pooled and getattr(self, 'rebalance_datasets', None) \
                        and getattr(self.rebalance_datasets, 'method', None) not in (None, 'none'):
                    logger.warning(
                        "[Rebalance] requested (method=%s) but only implemented for agg_level='night'. Skipping.",
                        self.rebalance_datasets.method
                    )

        x_train = np.array(x[train_index_mask])
        y_train = np.array(y[train_index_mask])
        y_str_train = np.array(y_str[train_index_mask]) if self.add_str_labels else None

        x_test = np.array(x[test_index_mask])
        y_test = np.array(y[test_index_mask])
        y_str_test = np.array(y_str[test_index_mask]) if self.add_str_labels else None

        self._assert_binary_labels(y_train, 'y_train')
        self._assert_binary_labels(y_test, 'y_test')
        class_weights_train = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train) \
            if y_train.shape[0] > 0 else None

        if self.verbose:
            counts_train = np.unique(y_train, return_counts=True)[1] if y_train.shape[0] > 0 else [0, 0]
            counts_test = np.unique(y_test, return_counts=True)[1] if y_test.shape[0] > 0 else [0, 0]
            logger.info(f"Selected features and aggregated DataFrame shape: {x.shape}")
            if y_train.shape[0] > 0:
                train_rbd_percent = counts_train[1] / y_train.shape[0] * 100
                train_hc_percent = counts_train[0] / y_train.shape[0] * 100
            else:
                train_rbd_percent = train_hc_percent = 0
            if y_test.shape[0] > 0:
                test_rbd_percent = counts_test[1] / y_test.shape[0] * 100
                test_hc_percent = counts_test[0] / y_test.shape[0] * 100
            else:
                test_rbd_percent = test_hc_percent = 0
            logger.info(f"Total {agg_level} instances in train set: {y_train.shape[0]}\n"
                        f"\t - RBD: {counts_train[1]:6.0f} = {train_rbd_percent:4.1f}% "
                        f"\t - HC:  {counts_train[0]:6.0f} = {train_hc_percent:4.1f}%")

            logger.info(f"Total {agg_level} instances in test set: {y_test.shape[0]}\n"
                        f"\t - RBD: {counts_test[1]:6.0f} = {test_rbd_percent:4.1f}% "
                        f"\t - HC:  {counts_test[0]:6.0f} = {test_hc_percent:4.1f}%")

        # check for NaNs
        _datasets = {'x_train': x_train, 'y_train': y_train, 'x_test': x_test, 'y_test': y_test}
        nan_summary = []
        for name, data in _datasets.items():
            nan_count = np.isnan(data).sum()
            nan_summary.append(f'{name}: {nan_count} NaNs')
        if any(np.isnan(data).sum() > 0 for data in _datasets.values()):
            logging.warning('found NaNs in data: - ' + ', '.join(nan_summary))
        else:
            if self.verbose:
                logging.info('no NaNs in x_train, y_train, x_test, y_test.\n')
        train_dataset_ids = np.array([self.map_subject_to_dataset[_id] for _id in self.train_id_map]) \
            if self.dataset_is_pooled else None
        test_dataset_ids = np.array([self.map_subject_to_dataset[_id] for _id in self.test_id_map]) \
            if self.dataset_is_pooled else None

        train_set = FeatureSet(
            x=x_train, y=y_train, y_str=y_str_train,  # data and labels
            group=self.train_id_map, feat_map=feat_map, dataset=train_dataset_ids  # mappings
        )
        test_set = FeatureSet(
            x=x_test, y=y_test, y_str=y_str_test,
            group=self.test_id_map, feat_map=feat_map, dataset=test_dataset_ids
        )

        return train_set, test_set, class_weights_train

    @staticmethod
    def _check_and_filter_feature_files(local_feat_files: List[Path], meta_df: pd.DataFrame) -> List[Path]:
        """ 1) Warn if any expected feature files (according to meta_df) are missing.
            2) Warn if there are any extra feature files not in the filtered meta_df.
            3) Return only the files matching the meta_df entries."""
        # build expected suffixes
        expected = set()
        for _, row in meta_df.iterrows():
            _id = row['ID']
            rid = str(row.get('record_ID')).strip()
            suf = _id if pd.isna(row.get('record_ID')) or rid.lower() in {'none', ''} else f"{_id}_{rid}"
            expected.add(suf)

        # map each actual file to its suffix
        suffix_to_paths = defaultdict(list)
        for p in local_feat_files:
            suf = p.stem.replace("local-features-", "")
            suffix_to_paths[suf].append(p)
        actual = set(suffix_to_paths.keys())

        missing, extra = expected - actual, actual - expected
        if missing:
            logger.warning(f"Missing {len(missing)} expected feature file(s): {sorted(missing)}")
        if extra:
            logger.debug(
                f"Found {len(extra)} unexpected feature file(s) (excluded or not in meta e.g. because of unmapped "
                f"diagnosis): {sorted(extra)}")

        # filter the original list, keeping only paths whose suffix is expected
        filtered = [p for suf, paths in suffix_to_paths.items() if suf in expected for p in paths]
        return sorted(filtered)

    @staticmethod
    def _assert_binary_labels(y_array, name='y_array'):
        if y_array.size > 0 and not np.array_equal(np.unique(y_array), [0, 1]):
            raise AssertionError(f"{name} contains non-binary labels: {np.unique(y_array)}")

    def _get_meta(self, meta_csv_path: Path):
        # read meta file

        meta_df = utils.read_meta_csv_to_df(meta_csv_path, exclude=True, verbose=self.verbose)

        assert 'diagnosis' in meta_df.columns, f"'meta_df' at {meta_csv_path} must contain a 'diagnosis' column."
        _is_null = meta_df.diagnosis.isnull()
        assert not meta_df.diagnosis.isnull().any(), "'diagnosis' column contains NaN values."

        mapping_keys, data_keys = set(self.binary_mapping.keys()), set(meta_df['diagnosis'].unique())
        unused_keys = mapping_keys - data_keys
        if unused_keys:  # keys defined in mapping but not present in the data
            logger.warning(f"Mapping keys not present in this dataset: {unused_keys}")
        unmapped_keys = data_keys - mapping_keys
        if unmapped_keys:  # Keys present in the data but not covered by the mapping
            logger.warning(f"Dataset contains diagnoses not covered by mapping: {unmapped_keys}")
        # choose only relevant rows according to binary_mapping, will drop all other rows
        orig_counts = meta_df['diagnosis'].value_counts().to_dict()
        if self.verbose:
            logger.info(f"Original diagnoses (pre‐filter): {orig_counts}")
        keep_mask = meta_df['diagnosis'].isin(self.binary_mapping.keys())
        dropped_diags = meta_df.loc[~keep_mask, 'diagnosis'].value_counts().to_dict()
        if self.verbose:
            logger.info(f"Dropping {len(meta_df) - keep_mask.sum()} rows with unmapped diagnoses: {dropped_diags}")

        meta_df = meta_df[keep_mask]
        mapped_counts = meta_df['diagnosis'].value_counts().to_dict()
        if self.verbose:
            logger.info(f"Kept {len(meta_df)} rows; mapped diagnoses: {mapped_counts}")
        # create binary labels
        meta_df.insert(
            meta_df.columns.get_loc('diagnosis') + 1, 'binary_label', meta_df.diagnosis.map(self.binary_mapping))

        # get the ids of the training and testing records
        records_train = meta_df.loc[meta_df['train/test'] == 'train'].apply(
            lambda row: row['ID'] if pd.isna(row['record_ID']) else f"{row['ID']}_{str(row['record_ID']).strip()}",
            axis=1).to_numpy().astype('str')
        records_test = meta_df.loc[meta_df['train/test'] == 'test'].apply(
            lambda row: row['ID'] if pd.isna(row['record_ID']) else f"{row['ID']}_{str(row['record_ID']).strip()}",
            axis=1).to_numpy().astype('str')

        return meta_df, records_train, records_test

    @staticmethod
    def _get_class_counts(meta_df: pd.DataFrame):
        _unique_labels = meta_df.binary_label.unique()
        assert len(_unique_labels) == 2, f"only two unique labels allowed, got {_unique_labels}."
        if sorted(_unique_labels) == ['HC', 'RBD']:
            _hc_key, _rbd_key = 'HC', 'RBD'
        elif sorted(_unique_labels) == [0, 1]:
            _hc_key, _rbd_key = 0, 1
        else:
            raise ValueError(f"binary labels are not of expected format 'HC'/'RBD' or 0/1, got {_unique_labels}.")

        train_df = meta_df[meta_df['train/test'] == 'train']
        if not train_df.empty:
            _value_count_train = train_df.binary_label.value_counts()
            n_hc_train, n_rbd_train = _value_count_train.loc[_hc_key], _value_count_train.loc[_rbd_key]
        else:
            n_hc_train, n_rbd_train = 0, 0

        test_df = meta_df[meta_df['train/test'] == 'test']
        if not test_df.empty:
            _value_count_test = test_df.binary_label.value_counts()
            n_hc_test, n_rbd_test = _value_count_test.loc[_hc_key], _value_count_test.loc[_rbd_key]
        else:
            n_hc_test, n_rbd_test = 0, 0

        return n_hc_train, n_rbd_train, n_hc_test, n_rbd_test

    @staticmethod
    def _infer_dataset_dir_mapping(meta_df: pd.DataFrame, patient_dirs: List[Path]) -> dict:
        """Get mapping from dataset_id to patient_dir using file existence checks.
         Allows multiple dataset_ids to share the same directory."""
        dir_to_dataset_ids = defaultdict(set)
        for p_dir in patient_dirs:
            for _, row in meta_df.iterrows():
                full_path = p_dir / row['ID']
                if full_path.exists():
                    dir_to_dataset_ids[p_dir].add(row['dataset_id'])

        patient_dir_to_dataset_id = {}
        for p_dir, ds_ids in dir_to_dataset_ids.items():
            if len(ds_ids) != 1:
                logger.warning(f"Directory {p_dir} maps to multiple dataset_ids: {ds_ids}")
            for ds_id in ds_ids:
                patient_dir_to_dataset_id[ds_id] = p_dir

        mapped_dataset_ids = set(patient_dir_to_dataset_id.keys())
        all_dataset_ids = set(meta_df['dataset_id'].unique())

        if mapped_dataset_ids != all_dataset_ids:
            missing = all_dataset_ids - mapped_dataset_ids
            raise ValueError(f"No matching patient_dir found for dataset_ids: {missing}")

        return patient_dir_to_dataset_id

    def _handle_problematic_values(self, df, df_log_name: str, drop: bool, replace: dict, excluded_cols: List[str]):
        """ Handles problematic values in a DataFrame by either dropping the rows or replacing the values.

        Parameters:
            :param df: (pd.DataFrame) The DataFrame to process.
            :param df_log_name: (str) The name of the DataFrame for logging purposes.
            :param drop: (bool) If True, drop rows with problematic values.
            :param replace: (dict) A dictionary mapping problematic value types to their replacement values.
                            Example: {'NaN': 0, 'Inf': 999, 'Whitespace': 'NA', 'Non-Standard Missing': 'NA'}

        Returns:
            :return: (pd.DataFrame) The DataFrame with problematic values handled as specified."""
        assert not (drop and replace), "specify either 'drop' or 'replace', but not both."

        numerical_cols = df.select_dtypes(include=[np.number]).columns.difference(excluded_cols)

        def _val_is_problematic(value):  # check whether value is problematic or not, i.e. NaN, Inf, whitespace, etc.
            try:
                if pd.isna(value):
                    return 'NaN'
                elif np.isinf(value):
                    return 'Inf'
                elif value == '' or value == ' ':
                    return 'Whitespace'
                elif value in ['N/A', 'null']:
                    return 'Non-Standard Missing'
                else:
                    return None
            except TypeError:
                return None

        # is_problematic_df = df.applymap(_val_is_problematic)
        # problematic_counts = is_problematic_df.stack().value_counts()
        is_problematic_df = df[numerical_cols].map(_val_is_problematic)
        problematic_counts = is_problematic_df.stack().value_counts()

        if self.verbose:
            total_cells = df[numerical_cols].size
            total_problematic = int(problematic_counts.sum())
            frac = total_problematic / total_cells if total_cells else 0.0
            if total_problematic > 0:
                detail = ", ".join(f"{typ}: {cnt}" for typ, cnt in problematic_counts.items())
                logger.info(
                    f"found {total_problematic} problematic cells ({frac * 100:.1g}% of "
                    f"{numerical_cols.size} cols×{df.shape[0]} rows) in '{df_log_name}': {detail}. "
                    f"Using drop={drop}, replace={replace}.")
            else:
                logger.info(f"found no problematic values in '{df_log_name}' DataFrame.")

        if drop:  # drop the problematic rows
            df = df[~is_problematic_df.notna().any(axis=1)]
        elif replace:  # replace with pre-defined replacement values
            def _replace_val(val):
                problem_type = _val_is_problematic(val)
                if problem_type in replace:
                    return replace[problem_type]
                return val

            df[numerical_cols] = df[numerical_cols].applymap(_replace_val)

        return df

    def _create_local_feature_dataframe(self, util_cols: List[str] = None):
        if self.verbose:
            logger.info('creating local feature DataFrame... ')
        util_cols = util_cols if util_cols else [
            'id', 'record_id', 'diagnosis', 'time_start', 'time_end', 'time_diff', 'sptw_start',
            'sptw_end', 'sptw_idx', 'night', 'runtime', 'ident']

        with utils.Timer() as timer:
            _li = []

            for filename in self.local_feat_files:
                _df = pd.read_csv(filename, index_col=None, header=0)
                _df = _df.loc[:, ~_df.columns.str.contains('Unnamed')]  # drop all columns with 'Unnamed' (old indices)
                _df['ident'] = filename
                if not _df.empty:
                    _record_id_missing = 'record_id' not in _df.columns
                    if _record_id_missing:
                        _df['record_id'] = None

                    _df = _df[self.included_local_features + util_cols]
                    _df['record_id'] = _df['record_id'].astype(str).str.strip()
                    _df['record_id'] = _df['record_id'].replace(['none', 'NaN', 'nan', 'None'], None)
                    _df['record_key'] = _df.apply(lambda row: row['id'] if row['record_id'] is None
                    else f"{row['id']}_{row['record_id']}", axis=1)
                    _li.append(_df)

            _local_feat_df = pd.concat(_li, axis=0, ignore_index=True)
            _local_feat_df['month'] = pd.to_datetime(_local_feat_df.time_start).dt.month
            _local_feat_df['day'] = pd.to_datetime(_local_feat_df.time_start).dt.day
            _local_feat_df['hour'] = pd.to_datetime(_local_feat_df.time_start).dt.hour
            _local_feat_df['minute'] = pd.to_datetime(_local_feat_df.time_start).dt.minute

            # 'toilette' filter
            _local_feat_df = _local_feat_df[(_local_feat_df.time_diff < 50) & (_local_feat_df.time_diff > .5)]
            _local_feat_df['ground_truth'] = _local_feat_df.diagnosis.map(self.binary_mapping).values
            _local_feat_df = self._handle_problematic_values(
                _local_feat_df, drop=True, replace=None, df_log_name='local_feat_df', excluded_cols=util_cols)

        if self.verbose:
            logger.info(f'creating local feature DataFrame... done. ({timer()}s)')
        return _local_feat_df

    def _log_info_(self, n_rbd_train, n_hc_train, n_rbd_test, n_hc_test):
        logger.info(
            f"full dataset contains n = {self.num_total} patients:\n"
            f"\t\t - n_rbd = {self.num_total_rbd} ({self.num_total_rbd / self.num_total * 100:.1f}%)\n"
            f"\t\t - n_hc  = {self.num_total_hc} ({self.num_total_hc / self.num_total * 100:.1f}%)"
        )
        logger.info(
            f"train/test split:\n"
            f"\t\t - train: n = {n_rbd_train + n_hc_train:2.0f},"
            f"rbd = {n_rbd_train:2.0f} "
            f"({n_rbd_train / (n_rbd_train + n_hc_train) * 100 if n_rbd_train + n_hc_train != 0 else 0:4.1f}%)"
            f" hc = {n_hc_train:2.0f} "
            f"({n_hc_train / (n_rbd_train + n_hc_train) * 100 if n_rbd_train + n_hc_train != 0 else 0:4.1f}%)\n"
            f"\t\t - test:  n = {n_rbd_test + n_hc_test:2.0f},"
            f"rbd = {n_rbd_test:2.0f} "
            f"({n_rbd_test / (n_rbd_test + n_hc_test) * 100 if n_rbd_test + n_hc_test != 0 else 0:4.1f}%)"
            f" hc = {n_hc_test:2.0f} "
            f"({n_hc_test / (n_rbd_test + n_hc_test) * 100 if n_rbd_test + n_hc_test != 0 else 0:4.1f}%)"
        )

        if self.dataset_is_pooled:
            logger.info("Dataset-wise contributions:")
            for dataset_id in sorted(self.meta_df['dataset_id'].unique()):
                df_subset = self.meta_df[self.meta_df['dataset_id'] == dataset_id]
                for split in ['train', 'test']:
                    df_split = df_subset[df_subset['train/test'] == split]
                    count_total = df_split.shape[0]
                    count_rbd = df_split[df_split['binary_label'] == 1].shape[0]
                    count_hc = df_split[df_split['binary_label'] == 0].shape[0]
                    logger.info(
                        f"\t - dataset {dataset_id} ({split}): total = {count_total:3d}, "
                        f"rbd = {count_rbd:3d}, hc = {count_hc:3d}, "
                        f"rbd = {count_rbd / count_total * 100 if count_total > 0 else 0:4.1f}%, "
                        f"hc = {count_hc / count_total * 100 if count_total > 0 else 0:4.1f}%"
                    )

    def _aggregate_local_to_global(self, _local_df_copy: pd.DataFrame, use_numba: bool):

        if use_numba:
            @jit(nopython=True)
            def mad_numba(x):
                mean_x = np.mean(x)
                return np.mean(np.abs(x - mean_x))

            @jit(nopython=True)
            def iqr_numba(x):
                return np.percentile(x, 75) - np.percentile(x, 25)

            @jit(nopython=True)
            def percentile_10_numba(x):
                return np.percentile(x, 10)

            @jit(nopython=True)
            def percentile_90_numba(x):
                return np.percentile(x, 90)

            aggregation_functions = {
                'mean': 'mean',
                'std': 'std',
                'median': 'median',
                'mad': ('mad', lambda x: mad_numba(x.values)),
                'skew': ('skew', lambda x: x.skew()),  # numba implementation using numpy was wrong
                'kurt': ('kurt', lambda x: x.kurt()),  # numba implementation using numpy was wrong
                'iqr': ('iqr', lambda x: iqr_numba(x.values)),
                '10th_percentile': ('10th_percentile', lambda x: percentile_10_numba(x.values)),
                '90th_percentile': ('90th_percentile', lambda x: percentile_90_numba(x.values))
            }

        else:
            aggregation_functions = {
                'mean': 'mean',
                'std': 'std',
                'median': 'median',
                'mad': ('mad', lambda x: (x - x.mean()).abs().mean()),
                'skew': ('skew', lambda x: x.skew()),
                'kurt': ('kurt', lambda x: x.kurt()),
                'iqr': ('iqr', lambda x: x.quantile(.75) - x.quantile(.25)),
                '10th_percentile': ('10th_percentile', lambda x: x.quantile(.1)),
                '90th_percentile': ('90th_percentile', lambda x: x.quantile(.9))
            }

        reducer = {feature: [aggregation_functions[method] for method in self.aggregation]
                   for feature in self.included_local_features}

        def _unique_first(x):
            return x.iloc[0]

        reducer['ground_truth'] = _unique_first  # unique ints, choose first
        reducer['diagnosis'] = _unique_first  # unique strings, choose first
        reducer['id'] = _unique_first  # unique strings, choose first
        reducer['record_id'] = _unique_first  # unique strings, choose first

        with utils.Timer() as agg_timer:
            global_feat_df = _local_df_copy.groupby(['record_key', 'night'])[
                self.included_local_features + ['id', 'record_id', 'ground_truth', 'diagnosis']
                ].agg(reducer).reset_index()
            if self.verbose:
                logger.info(f"local to global aggregation done. ({agg_timer()}s)")
        global_feat_df.columns = ['_'.join(col).strip() if col[1] else col[0] for col in global_feat_df.columns]
        global_feat_df.rename(columns={'ground_truth__unique_first': 'ground_truth'}, inplace=True)
        global_feat_df.rename(columns={'diagnosis__unique_first': 'diagnosis'}, inplace=True)
        global_feat_df.rename(columns={'id__unique_first': 'id'}, inplace=True)
        global_feat_df.rename(columns={'record_id__unique_first': 'record_id'}, inplace=True)

        return self._handle_problematic_values(
            global_feat_df, drop=True, replace=None, df_log_name='global_feat_df',
            excluded_cols=['record_key', 'id', 'record_id', 'diagnosis', 'ground_truth'])

    def _create_global_feature_dataframe(self, local_df_copy: pd.DataFrame, use_numba: bool = True,
                                         return_structure_only: bool = False, global_only: bool = False):

        if return_structure_only:
            return local_df_copy[
                ['id', 'record_id', 'record_key', 'night', 'ground_truth']].drop_duplicates().reset_index(drop=True)

        if global_only:  # only get skeleton
            global_feat_df = local_df_copy[
                ['id', 'record_id', 'record_key', 'night', 'ground_truth', 'diagnosis']
            ].drop_duplicates().reset_index(drop=True)
        else:  # aggregate local features
            global_feat_df = self._aggregate_local_to_global(local_df_copy, use_numba=use_numba)

        for _record_key, _id, _record_id in zip(
                global_feat_df.record_key.unique(), global_feat_df.groupby('record_key')['id'].first(),
                global_feat_df.groupby('record_key')['record_id'].first()):

            # glob the global feature files for each record
            if self.dataset_is_pooled:
                _dataset_id = self.map_subject_to_dataset[_id]
                patient_dir = self.map_dataset_to_data_dir[_dataset_id]
            else:
                patient_dir = self.data_dirs[0]

            _feature_dir_path = (patient_dir / f"{_id}/{self.feature_dir}" if _record_id in [None, 'None']
                                 else patient_dir / f"{_id}" / f"{_id}_{_record_id}/{self.feature_dir}")
            _global_feat_paths = sorted(list(_feature_dir_path.glob('*global*.csv')))

            if len(_global_feat_paths) != 1:
                raise UserWarning(f"none or too many global feature files found for {_record_key}.")

            else:
                _global_features = pd.read_csv(_global_feat_paths[0])
                _global_features = _global_features[self.included_global_features + ['id', 'night']]

                global_feat_df = pd.merge(
                    global_feat_df, _global_features, on=['id', 'night'], how='left', suffixes=('', '_dup'))

                for col in _global_features.columns.difference(['id', 'night']):
                    dup_col = f'{col}_dup'
                    if dup_col in global_feat_df.columns:
                        global_feat_df[col] = global_feat_df[col].fillna(global_feat_df.pop(dup_col))
        return global_feat_df

    def _apply_dataset_rebalancing(
            self, df_night: pd.DataFrame,  # aggregated night-level DF (has 'record_key','id','ground_truth',...)
            train_index_mask: np.ndarray,  # boolean mask over df_night rows
    ) -> np.ndarray:
        """Rebalance the *training* portion by downsampling records per dataset.
        Operates at record level and then keeps all nights from the chosen records.
        Supported methods (self.rebalance_datasets.method):
          - 'none' / None: do nothing
          - 'min': equalize all datasets to the size of the smallest (strict downsample)
          - 'median': cap each dataset at the median dataset size
          - 'cap_absolute': cap each dataset at a fixed maximum (uses .max_per_dataset)
          - 'cap_to_second_largest': cap the largest dataset to the size of the
             second largest if it exceeds dominance_ratio × second_largest
             (uses .dominance_ratio; default 1.4–1.5 is sensible)
        If .preserve_class_ratio is True, per-dataset positive/negative sampling
        follows the dataset’s original train-split class proportions."""
        cfg = getattr(self, "rebalance_datasets", None)
        if not cfg or not getattr(cfg, "method", None) or cfg.method in ("none", None):
            return train_index_mask

        assert self.dataset_is_pooled, \
            "Rebalancing by dataset requires pooled mode with 'dataset_id' available."

        # ---------------------------------------------
        # Build a RECORD-LEVEL view of the *train* part
        # ---------------------------------------------
        train_df = df_night.loc[train_index_mask, ["record_key", "id", "ground_truth"]].copy()

        # one row per record_key (record = subject or subject_record)
        rec_df = (
            train_df.groupby("record_key")
            .agg(id=("id", "first"),
                 y=("ground_truth", "first"))
            .reset_index()
        )

        # map to dataset_id
        rec_df["dataset_id"] = rec_df["id"].map(self.map_subject_to_dataset)
        if rec_df["dataset_id"].isna().any():
            missing = rec_df.loc[rec_df["dataset_id"].isna(), "id"].unique().tolist()
            raise ValueError(f"Missing dataset_id mapping for IDs: {missing[:5]}{'...' if len(missing) > 5 else ''}")

        rng = np.random.default_rng(getattr(cfg, "seed", 42))

        # ---------------------------------------------
        # Determine per-dataset target sizes
        # ---------------------------------------------
        sizes = rec_df.groupby("dataset_id").size().sort_values()
        ds_order = list(sizes.index)
        ds_sizes = sizes.to_dict()

        method = cfg.method
        target_sizes = {}

        if method == "min":
            target = int(sizes.min())
            for ds in ds_order:
                target_sizes[ds] = min(ds_sizes[ds], target)

        elif method == "median":
            target = int(np.median(sizes.values))
            for ds in ds_order:
                target_sizes[ds] = min(ds_sizes[ds], target)

        elif method == "cap_absolute":
            max_cap = int(getattr(cfg, "max_per_dataset", 0) or 0)
            if max_cap <= 0:
                raise ValueError("cap_absolute requires a positive 'max_per_dataset'.")
            for ds in ds_order:
                target_sizes[ds] = min(ds_sizes[ds], max_cap)

        elif method == "cap_to_second_largest":
            # cap ONLY the largest dataset if it's too dominant vs the 2nd largest
            if len(sizes) < 2:
                # nothing to do with a single dataset
                return train_index_mask
            second = int(sizes.iloc[-2])
            largest_ds = sizes.index[-1]
            largest_n = int(sizes.iloc[-1])
            ratio = float(getattr(cfg, "dominance_ratio", 1.5))

            for ds in ds_order:
                if ds == largest_ds and largest_n > ratio * second:
                    target_sizes[ds] = second
                else:
                    target_sizes[ds] = ds_sizes[ds]

        else:
            raise ValueError(f"Unknown rebalancing method: {method}")

        # ---------------------------------------------
        # Sample record_keys per dataset (optionally preserving class ratios)
        # ---------------------------------------------
        keep_record_keys = []

        preserve_ratio = bool(getattr(cfg, "preserve_class_ratio", True))
        for ds, target_n in target_sizes.items():
            sub = rec_df[rec_df["dataset_id"] == ds]
            if target_n >= len(sub):
                keep_record_keys.extend(sub["record_key"].tolist())
                continue

            if not preserve_ratio:
                sel = sub.sample(n=target_n, random_state=int(getattr(cfg, "seed", 42)))
                keep_record_keys.extend(sel["record_key"].tolist())
            else:
                # preserve within-dataset class mix
                pos = sub[sub["y"] == 1]
                neg = sub[sub["y"] == 0]
                n_pos = len(pos)
                n_neg = len(neg)
                if n_pos + n_neg == 0:
                    continue

                frac_pos = n_pos / (n_pos + n_neg)
                n_pos_target = int(round(frac_pos * target_n))
                n_neg_target = target_n - n_pos_target

                pos_keep = pos.sample(
                    n=min(n_pos_target, n_pos),
                    random_state=int(getattr(cfg, "seed", 42))
                )
                neg_keep = neg.sample(
                    n=min(n_neg_target, n_neg),
                    random_state=int(getattr(cfg, "seed", 42) + 1)
                )

                # If rounding / scarcity left us short, top up from the larger class
                short = target_n - (len(pos_keep) + len(neg_keep))
                if short > 0:
                    remainder = sub.drop(pos_keep.index.union(neg_keep.index), errors="ignore")
                    if len(remainder) > 0:
                        extra = remainder.sample(
                            n=min(short, len(remainder)),
                            random_state=int(getattr(cfg, "seed", 42) + 2)
                        )
                        take = pd.concat([pos_keep, neg_keep, extra], axis=0)
                    else:
                        take = pd.concat([pos_keep, neg_keep], axis=0)
                else:
                    take = pd.concat([pos_keep, neg_keep], axis=0)

                keep_record_keys.extend(take["record_key"].tolist())

        keep_record_keys = set(keep_record_keys)

        # ----------------------------------------------------
        # Convert the chosen records back to a row-wise mask
        # ----------------------------------------------------
        new_train_mask = train_index_mask.copy()
        train_rows = np.where(train_index_mask)[0]
        train_rec_keys = df_night.loc[train_index_mask, "record_key"].values
        keep_mask_local = np.array([rk in keep_record_keys for rk in train_rec_keys], dtype=bool)
        new_train_mask[train_rows] = keep_mask_local

        # Logging summary
        before = {ds: int(n) for ds, n in ds_sizes.items()}
        after = {
            ds: int((rec_df["dataset_id"].isin([ds]) & rec_df["record_key"].isin(keep_record_keys)).sum())
            for ds in ds_order
        }
        logger.info(f"[Rebalance] method={method}  preserve_class_ratio={preserve_ratio}")
        logger.info(f"[Rebalance] per-dataset counts (records): before={before}  →  after={after}")

        return new_train_mask

    def _log_rebalanced_train_summary(self, df_night: pd.DataFrame, train_index_mask: np.ndarray):
        """Log an info summary comparable to _log_info_ but for the TRAIN split after rebalancing.
           Works at record level (one row per record_key)."""
        # One row per record (record_key), with id, y, dataset_id
        rec_df = (
            df_night.loc[train_index_mask, ['record_key', 'id', 'ground_truth']]
            .drop_duplicates()
            .assign(
                dataset_id=lambda d: d['id'].map(self.map_subject_to_dataset) if self.dataset_is_pooled else 'SINGLE')
        )

        # Totals
        total = len(rec_df)
        n_rbd = int((rec_df['ground_truth'] == 1).sum())
        n_hc = total - n_rbd
        p_rbd = (n_rbd / total * 100) if total else 0.0
        p_hc = (n_hc / total * 100) if total else 0.0

        logger.info("[Rebalance][post] train split (records): n = %d", total)
        logger.info("[Rebalance][post] \t - n_rbd = %d (%.1f%%)", n_rbd, p_rbd)
        logger.info("[Rebalance][post] \t - n_hc  = %d (%.1f%%)", n_hc, p_hc)

        if self.dataset_is_pooled:
            logger.info("[Rebalance][post] Dataset-wise contributions (TRAIN):")
            for ds in sorted(rec_df['dataset_id'].unique()):
                sub = rec_df[rec_df['dataset_id'] == ds]
                t = len(sub)
                r = int((sub['ground_truth'] == 1).sum())
                h = t - r
                pr = (r / t * 100) if t else 0.0
                ph = (h / t * 100) if t else 0.0
                logger.info("\t - dataset %s (train): total = %3d, rbd = %3d, hc = %3d, rbd = %4.1f%%, hc = %4.1f%%",
                            ds, t, r, h, pr, ph)
