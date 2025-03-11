import logging
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numba import jit
from scipy import stats
from sklearn.utils.class_weight import compute_class_weight

from aktiRBD import utils
from aktiRBD.classifier.tools.feature_set import FeatureSet
from aktiRBD.classifier.tools.aggregation import dataframe_agg_night_2_patient

__all__ = ['DataLoader']

logger = logging.getLogger(__name__)


class DataLoader:

    # todo: add docstring

    def __init__(self, patient_dir: Path, meta_path: Union[Path, pd.DataFrame], feature_dir: Path, aggregation: str,
                 included_local_features: list, included_global_features: list, binary_mapping: dict,
                 add_str_labels: bool = False, verbose: bool = True, shuffle_ids: bool = True):

        self.patient_dir = patient_dir
        self.feature_dir = feature_dir
        self.local_feat_files = sorted(list(self.patient_dir.glob(f"*/{feature_dir}/*local*.csv")))
        if len(self.local_feat_files) == 0:
            if verbose:
                logger.warning(
                    f'no local feature files found at {self.patient_dir.joinpath(f"*/{feature_dir}")}, trying'
                    f' {self.patient_dir.joinpath(f"*/*/{feature_dir}")} )')
            self.local_feat_files = sorted(list(self.patient_dir.glob(f"*/*/{feature_dir}/*local*.csv")))
            if len(self.local_feat_files) == 0:
                raise UserWarning(f"could not locate local feature files make sure {self.patient_dir} exists and "
                                  f"contains patient subdirs populated with the feature files.")
        # todo: how to handle multiple files per ID!!

        self.aggregation = aggregation
        self.binary_mapping = binary_mapping
        _meta_df = self._get_meta(meta_path, binary_mapping)  # already filters patient groups

        self.ids_train = _meta_df.loc[_meta_df['train/test'] == 'train', 'ID'].to_numpy().astype('str')
        self.ids_test = _meta_df.loc[_meta_df['train/test'] == 'test', 'ID'].to_numpy().astype('str')

        if shuffle_ids:
            np.random.shuffle(self.ids_train)
            np.random.shuffle(self.ids_test)

        self.included_local_features = included_local_features
        self.included_global_features = included_global_features
        self.verbose = verbose
        self.add_str_labels = add_str_labels

        n_hc_train, n_rbd_train, n_hc_test, n_rbd_test = self._get_class_counts(_meta_df)

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
        _file_ids = local_feat_df.id.unique().astype('str')
        _feature_ids = np.unique([x.split('_')[0] for x in _file_ids])  # in case of multiple files per ID, file name
        # must follow convention '<ID>-<123>_<file_ident>': todo: might need more robust solution in the future

        _missing_ids = np.setdiff1d(np.concatenate([self.ids_train, self.ids_test]), _feature_ids)
        if len(_missing_ids) > 0:
            logger.warning(f"features of {len(_missing_ids)} cases are missing: {_missing_ids}")

        x, y, y_str, feat_map, train_index_mask, test_index_mask = None, None, None, None, None, None

        if agg_level == 'move':
            _feature_df = local_feat_df[self.included_local_features]
            feat_map = _feature_df.columns.values
            x = _feature_df.to_numpy()
            y = local_feat_df.ground_truth
            y_str = local_feat_df.diagnosis if self.add_str_labels else None
            test_index_mask = local_feat_df.id.isin(self.ids_test)
            train_index_mask = local_feat_df.id.isin(self.ids_train)

        else:

            feat_df_night = self._create_global_feature_dataframe(local_feat_df.copy(), use_numba=agg_with_numba)
            if agg_level == 'night':

                _included_local_features = [
                    f"{_feat}_{postfix}" for postfix in self.aggregation for _feat in self.included_local_features
                ]
                _feature_df = feat_df_night[_included_local_features + self.included_global_features]

                feat_map = _feature_df.columns.values
                x = _feature_df.to_numpy()
                y = feat_df_night.ground_truth
                y_str = feat_df_night.diagnosis if self.add_str_labels else None
                feat_df_night['base_id'] = feat_df_night['id'].apply(
                    lambda x: x.split('_')[0])  # file vs pat_id, see earlier comment
                train_index_mask = feat_df_night.base_id.isin(
                    self.ids_train)  # problem ids are not file_ids but patient ids!! (fix it)
                test_index_mask = feat_df_night.base_id.isin(self.ids_test)
                # mask needs to check whether the pat_id is in self.ids_train, so use .split('_')[0] again!
                self.train_id_map = feat_df_night[train_index_mask].id.to_numpy()
                self.test_id_map = feat_df_night[test_index_mask].id.to_numpy()

            elif agg_level == 'patient':

                _included_local_features = [
                    f"{_feat}_{postfix}" for postfix in self.aggregation for _feat in self.included_local_features
                ]
                feat_df_patient = (
                    feat_df_night
                    .groupby(['id'])[_included_local_features + self.included_global_features + ['ground_truth']]
                    .agg('mean')  # take mean or too much loss of information??
                    .reset_index()
                )
                _feature_df = feat_df_patient[_included_local_features + self.included_global_features]

                feat_map = _feature_df.columns.values
                x = _feature_df.to_numpy()
                y = feat_df_patient.ground_truth
                test_index_mask = feat_df_patient.id.isin(
                    self.ids_test)  # problem ids are not file_ids but patient ids!! (see workaround for night level)
                train_index_mask = feat_df_patient.id.isin(self.ids_train)
                self.train_id_map = feat_df_patient[train_index_mask].id.to_numpy()
                self.test_id_map = feat_df_patient[test_index_mask].id.to_numpy()

        x_train = np.array(x[train_index_mask])
        y_train = np.array(y[train_index_mask])
        y_str_train = np.array(y_str[train_index_mask]) if self.add_str_labels else None

        x_test = np.array(x[test_index_mask])
        y_test = np.array(y[test_index_mask])
        y_str_test = np.array(y_str[test_index_mask]) if self.add_str_labels else None

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
        return meta_df

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

    def _handle_problematic_values(self, df, df_log_name: str, drop: bool, replace: dict,
                                   print_debug_info: bool = False):
        """
        Handles problematic values in a DataFrame by either dropping the rows or replacing the values.

        Parameters:
            :param df: (pd.DataFrame) The DataFrame to process.
            :param df_log_name: (str) The name of the DataFrame for logging purposes.
            :param drop: (bool) If True, drop rows with problematic values.
            :param replace: (dict) A dictionary mapping problematic value types to their replacement values.
                            Example: {'NaN': 0, 'Inf': 999, 'Whitespace': 'NA', 'Non-Standard Missing': 'NA'}
            :param print_debug_info: (bool, default=False) If True, print rows with problematic values for debugging.

        Returns:
            :return: (pd.DataFrame) The DataFrame with problematic values handled as specified.
        """
        # todo: exclude diagnosis/ground_truth column from check... (need to handle cases where only prediction
        #  without labels should be feasible!!)
        assert not (drop and replace), "specify either 'drop' or 'replace', but not both."

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
        is_problematic_df = df.map(_val_is_problematic)
        problematic_counts = is_problematic_df.stack(future_stack=True).value_counts()

        if self.verbose:
            if not problematic_counts.empty:
                log_message = ", ".join([f"{problem}: {count}" for problem, count in problematic_counts.items()])
                logger.info(f"found problematic values in '{df_log_name}' dataframe: {log_message}."
                            f" Using drop={drop}, replace={replace}")
            else:
                logger.info(f"found no problematic values in '{df_log_name}' dataframe")

        if print_debug_info:  # for debugging
            rows_with_problematic_values = df[is_problematic_df.notna().any(axis=1)]

            if 'ident' not in df.columns:
                print("Debug Info: 'ident' column not found in the DataFrame.")
            else:
                # Boolean DataFrame indicating problematic values
                is_problematic = is_problematic_df.notna()

                # Group by 'ident' and sum problematic counts per column
                problematic_counts_per_ident = is_problematic.groupby(df['ident']).sum()

                # Initialize global counter
                from collections import defaultdict
                global_problematic_counts = defaultdict(int)

                # Display per-ident problematic counts and accumulate global counts
                for ident, row in problematic_counts_per_ident.iterrows():
                    problematic_columns = row[row > 0]
                    if not problematic_columns.empty:
                        print(f"Ident: {ident}")
                        for column, count in problematic_columns.items():
                            print(f"  {column}: {count} problematic value(s)")
                            global_problematic_counts[column] += count
                        print("-" * 40)

                # Display global summary
                if global_problematic_counts:
                    print("\nGlobal Summary of Problematic Values:")
                    for column, total in global_problematic_counts.items():
                        print(f"  {column}: {total} problematic value(s)")
                    print("-" * 40)
                else:
                    print("No problematic values found across all files.")

        if drop:  # drop the problematic rows
            df = df[~is_problematic_df.notna().any(axis=1)]
        elif replace:  # replace with pre-defined replacement values
            def _replace_val(val):
                problem_type = _val_is_problematic(val)
                if problem_type in replace:
                    return replace[problem_type]
                return val

            df = df.applymap(_replace_val)

        return df

    def _create_local_feature_dataframe(self):
        logger.info('creating local feature dataframe... ')

        with utils.Timer() as timer:
            _li = []
            _util_cols = ['id', 'diagnosis', 'time_start', 'time_end', 'time_diff', 'sptw_start', 'sptw_end',
                          'sptw_idx', 'night', 'runtime', 'ident']

            for filename in self.local_feat_files:
                _df = pd.read_csv(filename, index_col=None, header=0)
                _df = _df.loc[:, ~_df.columns.str.contains('Unnamed')]  # drop all columns with 'Unnamed' (old indices)
                _df['ident'] = filename
                if not _df.empty:
                    _df = _df[self.included_local_features + _util_cols]
                    _li.append(_df)

            _local_feat_df = pd.concat(_li, axis=0, ignore_index=True)

            _local_feat_df['month'] = pd.to_datetime(_local_feat_df.time_start).dt.month
            _local_feat_df['day'] = pd.to_datetime(_local_feat_df.time_start).dt.day
            _local_feat_df['hour'] = pd.to_datetime(_local_feat_df.time_start).dt.hour
            _local_feat_df['minute'] = pd.to_datetime(_local_feat_df.time_start).dt.minute

            # 'toilette filter'
            _local_feat_df = _local_feat_df[(_local_feat_df.time_diff < 50) & (_local_feat_df.time_diff > .5)]

            _local_feat_df['ground_truth'] = _local_feat_df.diagnosis.map(self.binary_mapping).values

            _local_feat_df = self._handle_problematic_values(
                _local_feat_df, drop=True, replace=None, df_log_name='local_feat_df', print_debug_info=False)

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

    def _aggregate_local_to_global(self, _local_df_copy: pd.DataFrame, use_numba: bool, do_some_tests: bool = False):

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

        with utils.Timer() as agg_timer:
            global_feat_df = (
                _local_df_copy.groupby(['id', 'night'])[self.included_local_features + ['ground_truth', 'diagnosis']]
                .agg(reducer).reset_index())
            logger.info(f"local to global aggregation done ({agg_timer()}s)")

        global_feat_df = self._handle_problematic_values(
            global_feat_df, drop=True, replace=None, df_log_name='global_feat_df')
        global_feat_df.columns = ['_'.join(col).strip() if col[1] else col[0] for col in global_feat_df.columns]

        global_feat_df.rename(columns={'ground_truth__unique_first': 'ground_truth'}, inplace=True)
        global_feat_df.rename(columns={'diagnosis__unique_first': 'diagnosis'}, inplace=True)
        if do_some_tests:
            example_feature = 'mean_mag'
            test_df_local = _local_df_copy[['id', 'night', example_feature]]
            test_df_global = test_df_local.groupby(['id', 'night']).agg(
                [np.mean, np.std, stats.skew, stats.kurtosis, np.median, stats.iqr])

            hc_data = test_df_global.loc[test_df_global.index.get_level_values('id').str.startswith('HC')]
            rbd_data = test_df_global.loc[test_df_global.index.get_level_values('id').str.startswith('RBD')]
            fig, axs = plt.subplots(2, 3, figsize=(15, 10))
            axs = axs.flatten()
            metrics = ['mean', 'std', 'skew', 'kurtosis', 'median', 'iqr']
            for i, metric in enumerate(metrics):
                hc_values = hc_data[example_feature][metric].values
                rbd_values = rbd_data[example_feature][metric].values
                _stat_dict = utils.independent_stat_significance_test(hc_values, rbd_values, ('hc', 'rbd'))
                text_str = '\n'.join((
                    ' Normality Tests (HC):',
                    "   Shapiro-Wilk:",
                    f"      {_stat_dict['normality']['hc']['shapiro']['stat']:.2f},"
                    f" p = {_stat_dict['normality']['hc']['shapiro']['p']:.1e}",
                    "   Kolmogorov-Smirnov:",
                    f"      {_stat_dict['normality']['hc']['kolmogorov']['stat']:.2f},"
                    f" p = {_stat_dict['normality']['hc']['kolmogorov']['p']:.1e}",
                    "   Anderson-Darling:",
                    f"      {_stat_dict['normality']['hc']['anderson']['stat']:.2f},"
                    f" thres(5%) = {_stat_dict['normality']['hc']['anderson']['thres_5']:.2f}",
                    '',
                    ' Normality Tests (RBD):',
                    "   Shapiro-Wilk:",
                    f"      {_stat_dict['normality']['rbd']['shapiro']['stat']:.2f},"
                    f" p = {_stat_dict['normality']['rbd']['shapiro']['p']:.1e}",
                    "   Kolmogorov-Smirnov:",
                    f"      {_stat_dict['normality']['rbd']['kolmogorov']['stat']:.2f},"
                    f" p = {_stat_dict['normality']['rbd']['kolmogorov']['p']:.1e}",
                    "   Anderson-Darling:",
                    f"      {_stat_dict['normality']['rbd']['anderson']['stat']:.2f},"
                    f" thres(5%) = {_stat_dict['normality']['rbd']['anderson']['thres_5']:.2f}",
                    '',
                    ' Difference Tests:',
                    '   T-test:',
                    f"      {_stat_dict['difference']['T-test']['stat/z']:.2f},"
                    f" p: {_stat_dict['difference']['T-test']['p']:.1e},"
                    f" r: {_stat_dict['difference']['T-test']['r']:.2f}",
                    '   Mann-Whitney-U:',
                    f"     {_stat_dict['difference']['Mann-Whitney-U']['stat']:.2f}"
                    f"    p: {_stat_dict['difference']['Mann-Whitney-U']['p']:.1e},",
                    f"     z: {_stat_dict['difference']['Mann-Whitney-U']['z']:.2f},"
                    f"    r: {_stat_dict['difference']['Mann-Whitney-U']['r']:.2f}"
                ))

                # Add text box
                props = dict(boxstyle='round', facecolor='whitesmoke', alpha=.5)
                axs[i].text(
                    .3, .6,
                    text_str,
                    transform=axs[i].transAxes,
                    fontsize=7,
                    verticalalignment='center',
                    horizontalalignment='left',
                    bbox=props
                )

                axs[i].scatter(['HC'] * len(hc_values), hc_values,
                               label=f"HC: {np.mean(hc_values):.3f} ±{np.std(hc_values):.3f}", alpha=0.5, c='b')
                axs[i].scatter(['RBD'] * len(rbd_values), rbd_values,
                               label=f"RBD: {np.mean(rbd_values):.3f} ±{np.std(rbd_values):.3f}", alpha=0.5, c='r')
                axs[i].set_title(f'{metric} for HC vs RBD')
                axs[i].set_xlabel('Group')
                axs[i].set_ylabel(f'{metric}')
                axs[i].legend(loc=8)

            plt.tight_layout()
            fig.savefig('/Users/david/Desktop/test.pdf')

            plt.show()

            example_ids = ['HC-01', 'HC-10', 'HC-22', 'RBD-04', 'RBD-22', 'RBD-55']
            night = 2

            # test correlation to label for different distribution modes and RBD vs HC seperation
            for _id in example_ids:
                _id_df = _local_df_copy[_local_df_copy.id == _id]

                values = _id_df[_id_df.night == night][example_feature].values
                fig, ax = plt.subplots(1, 1)
                ax.set_title(f"{_id}, {example_feature}")
                ax.hist(values, color='b', bins=20)
                ax.axvline(np.mean(values), c='r')
                text_str = '\n'.join((
                    f" mean: {np.mean(values):.3f}",
                    f" std: {np.std(values):.3f}",
                    f" median: {np.median(values):.3f}",
                    f" skew: {stats.skew(values):.3f}",
                    f" kurt: {stats.kurtosis(values):.3f}",
                    f" iqr: {stats.iqr(values):.3f}",
                ))

                # Add text box
                props = dict(boxstyle='round', facecolor='whitesmoke', alpha=.5)
                ax.text(.7, .8, text_str,
                        transform=ax.transAxes,
                        fontsize=7,
                        verticalalignment='center',
                        horizontalalignment='left',
                        bbox=props
                        )

                plt.show()
                plt.close()

        return global_feat_df

    def _create_global_feature_dataframe(self, local_df_copy: pd.DataFrame, use_numba: bool = True,
                                         return_structure_only: bool = False, global_only: bool = False):
        if return_structure_only:
            return local_df_copy[['id', 'night', 'ground_truth']].drop_duplicates().reset_index(drop=True)

        if global_only:  # only get skeleton
            global_feat_df = \
                local_df_copy[['id', 'night', 'ground_truth', 'diagnosis']].drop_duplicates().reset_index(drop=True)
        else:  # aggregate local features
            global_feat_df = self._aggregate_local_to_global(local_df_copy, use_numba=use_numba)

        for _file_id in global_feat_df.id.unique():
            if len(_file_id.split('_')) > 1:  # in case of multiple files per patient ID,
                # todo: more robust solution, maybe add pat_id & file_id column to feature csv's!
                _pat_id = _file_id.split('_')[0]
                _pat_path = f"{_pat_id}/{_file_id}"
            else:
                _pat_id = _file_id
                _pat_path = f"{_pat_id}"

            _global_feat_paths = sorted(list(
                self.patient_dir.joinpath(f"{_pat_path}/{self.feature_dir}/").glob('*global*.csv')))

            if len(_global_feat_paths) != 1:
                raise UserWarning(f"none or too many global feature files found for {_file_id}.")

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

    def aggregate_night_predictions(self, y_pred_train_night, y_pred_test_night, y_prob_train_night, y_prob_test_night,
                                    n_max_nights: int, mean_prob_threshold: float, majority_vote_frac: int = 2):

        raise DeprecationWarning('not used anymore afaik')

        global_feat_df = self._create_global_feature_dataframe(
            self._create_local_feature_dataframe(),  # works but is ugly! -> make local_df property again?
            use_numba=True,
            return_structure_only=True)

        train_index_mask = global_feat_df.id.isin(self.ids_train)
        test_index_mask = global_feat_df.id.isin(self.ids_test)

        per_night_train = global_feat_df.loc[train_index_mask, ['id', 'night', 'ground_truth']]
        per_night_train['prob'] = y_prob_train_night
        per_night_train['pred'] = y_pred_train_night

        # only use first 7 nights: (a lot of patients did not do last night) (why exclude actually?)
        per_night_train = per_night_train[per_night_train.night <= n_max_nights - 1]  # -1: night counter starts at 0

        per_patient_train = dataframe_agg_night_2_patient(
            per_night_train, mean_prob_threshold=mean_prob_threshold, majority_vote_frac=majority_vote_frac)

        per_night_test = global_feat_df.loc[test_index_mask, ['id', 'night', 'ground_truth']]
        per_night_test['prob'] = y_prob_test_night
        per_night_test['pred'] = y_pred_test_night
        per_night_test = per_night_test[per_night_test.night <= n_max_nights - 1]  # -1: night counter starts at 0

        per_patient_test = dataframe_agg_night_2_patient(
            per_night_test, mean_prob_threshold=mean_prob_threshold, majority_vote_frac=majority_vote_frac)

        return per_patient_train, per_patient_test
