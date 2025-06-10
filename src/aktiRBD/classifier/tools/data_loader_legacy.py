import logging
from pathlib import Path
from typing import Union, List

import numpy as np
import pandas as pd
from numba import jit
from sklearn.utils.class_weight import compute_class_weight

from aktiRBD import utils
from aktiRBD.classifier.tools.feature_set import FeatureSet

__all__ = ['DataLoader']

logger = logging.getLogger(__name__)


class DataLoader:

    """Manages the retrieval of local and global feature files, processes metadata,
    applies feature aggregation strategies, and prepares train/test datasets."""

    def __init__(self, patient_dir: Path, meta_path: Union[Path, pd.DataFrame], feature_dir: Path, aggregation: str,
                 included_local_features: list, included_global_features: list, binary_mapping: dict,
                 add_str_labels: bool = False, verbose: bool = True, shuffle: bool = True):

        self.patient_dir = patient_dir
        self.feature_dir = feature_dir
        self.local_feat_files = sorted(list(self.patient_dir.glob(f"*/{feature_dir}/*local*.csv"))  # single record
                                       + list(self.patient_dir.glob(f"*/*/{feature_dir}/*local*.csv")))  # multi record
        if len(self.local_feat_files) == 0:
            raise UserWarning(f"could not locate local feature files make sure '{self.patient_dir}' exists and "
                              f"contains patient subdirs populated with the feature files.")

        self.aggregation = aggregation
        self.binary_mapping = binary_mapping
        self.meta_df, self.records_train, self.records_test = self._get_meta(meta_path, binary_mapping)
        self._check_missing_feature_files(self.local_feat_files, self.meta_df)
        if shuffle:
            np.random.shuffle(self.records_train)
            np.random.shuffle(self.records_test)

        self.included_local_features = included_local_features
        self.included_global_features = included_global_features
        self.verbose = verbose
        self.add_str_labels = add_str_labels

        n_hc_train, n_rbd_train, n_hc_test, n_rbd_test = self._get_class_counts(self.meta_df)

        self.train_id_map = self.test_id_map = None
        self.num_total_rbd = n_rbd_train + n_rbd_test
        self.num_total_hc = n_hc_train + n_hc_test
        self.num_total = self.num_total_rbd + self.num_total_hc
        if verbose:
            self._log_info_(n_rbd_train, n_hc_train, n_rbd_test, n_hc_test)

    def __str__(self):
        return f"DataLoader(n_rbd={self.num_total_rbd}, n_hc={self.num_total_hc})"

    def get_train_test_data(self, agg_level: str, agg_with_numba: bool = True):

        local_feat_df = self._create_local_feature_dataframe()
        x, y, y_str, feat_map, train_index_mask, test_index_mask = None, None, None, None, None, None

        if agg_level == 'move':
            _feature_df = local_feat_df[self.included_local_features]
            feat_map = _feature_df.columns.values
            x = _feature_df.to_numpy()
            y = local_feat_df.ground_truth
            y_str = local_feat_df.diagnosis if self.add_str_labels else None
            test_index_mask = local_feat_df.record_key.isin(self.records_train)
            train_index_mask = local_feat_df.record_key.isin(self.records_test)

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
                test_index_mask = feat_df_patient.record_key.isin(self.records_train)
                train_index_mask = feat_df_patient.id.isin(self.records_test)
                self.train_id_map = feat_df_patient[train_index_mask].id.to_numpy()
                self.test_id_map = feat_df_patient[test_index_mask].id.to_numpy()

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
            logging.info('no NaNs in x_train, y_train, x_test, y_test.\n')

        train_set = FeatureSet(x=x_train, y=y_train, y_str=y_str_train, group=self.train_id_map, feat_map=feat_map)
        test_set = FeatureSet(x=x_test, y=y_test, y_str=y_str_test, group=self.test_id_map, feat_map=feat_map)

        return train_set, test_set, class_weights_train

    @staticmethod
    def _check_missing_feature_files(local_feat_files: List, meta_df: pd.DataFrame):
        """ Checks that all entries in the metadata have an existing local feature file. Logs a warning if missing. """
        expected_record = set()
        for _, _row in meta_df.iterrows():
            _id = _row.get('ID')
            _record_id = str(_row.get('record_ID')).strip() if pd.notna(_row.get('record_ID')) else None
            _save_suffix = _id if not _record_id else f"{_id}_{_record_id}"
            expected_record.add(_save_suffix)
        actual_records = {file.stem.replace("local-features-", "") for file in local_feat_files}
        missing_records = expected_record - actual_records  # Find missing expected files
        if missing_records:
            logger.warning(f"Missing {len(missing_records)} expected feature files of records: {missing_records}")

    @staticmethod
    def _assert_binary_labels(y_array, name='y_array'):
        if y_array.size > 0 and not np.array_equal(np.unique(y_array), [0, 1]):
            raise AssertionError(f"{name} contains non-binary labels: {np.unique(y_array)}")

    @staticmethod
    def _get_meta(meta_csv_path: Path, binary_mapping: dict):
        # read meta file
        meta_df = utils.read_meta_csv_to_df(meta_csv_path).query('exclude != 1')
        assert 'diagnosis' in meta_df.columns, f"'meta_df' at {meta_csv_path} must contain a 'diagnosis' column."
        assert not meta_df['diagnosis'].isnull().any(), "'diagnosis' column contains NaN values."
        assert set(binary_mapping.keys()).issubset(set(meta_df['diagnosis'].unique())), \
            (f"Mapping values {set(binary_mapping.keys())} are not a subset of "
             f"diagnosis values {set(meta_df['diagnosis'].unique())}.")
        # choose only relevant rows according to binary_mapping
        meta_df = meta_df[meta_df.diagnosis.isin(binary_mapping.keys())]
        # create binary labels
        meta_df.insert(meta_df.columns.get_loc('diagnosis') + 1, 'binary_label', meta_df.diagnosis.map(binary_mapping))

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
            if not problematic_counts.empty:
                log_message = ", ".join([f"{problem}: {count}" for problem, count in problematic_counts.items()])
                logger.info(f"found problematic values in '{df_log_name}' dataframe: {log_message}."
                            f" Using drop={drop}, replace={replace}")
            else:
                logger.info(f"found no problematic values in '{df_log_name}' dataframe")

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
        logger.info('creating local feature dataframe... ')
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
                    _df['record_key'] = _df.apply(lambda row: row['id'] if row['record_id'] is None \
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

        logger.info(f'creating local feature dataframe... done. ({timer()}s)')
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
            f" rbd = {n_rbd_train:2.0f} "
            f"({n_rbd_train / (n_rbd_train + n_hc_train) * 100 if n_rbd_train + n_hc_train != 0 else 0:4.1f}%)"
            f"  hc = {n_hc_train:2.0f} "
            f"({n_hc_train / (n_rbd_train + n_hc_train) * 100 if n_rbd_train + n_hc_train != 0 else 0:4.1f}%)\n"
            f"\t\t - test:  n = {n_rbd_test + n_hc_test:2.0f},"
            f" rbd = {n_rbd_test:2.0f} "
            f"({n_rbd_test / (n_rbd_test + n_hc_test) * 100 if n_rbd_test + n_hc_test != 0 else 0:4.1f}%)"
            f"  hc = {n_hc_test:2.0f} "
            f"({n_hc_test / (n_rbd_test + n_hc_test) * 100 if n_rbd_test + n_hc_test != 0 else 0:4.1f}%)"
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
            logger.info(f"local to global aggregation done ({agg_timer()}s)")
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
            _feature_dir_path = self.patient_dir.joinpath(f"{_id}/{self.feature_dir}") if _id == _record_key \
                else self.patient_dir.joinpath(f"{_id}/{_record_id}/{self.feature_dir}")  # for multi-records per id

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

