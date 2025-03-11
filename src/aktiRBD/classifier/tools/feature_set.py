import logging
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, Optional, Union

import numpy as np
from sklearn import preprocessing

from aktiRBD.classifier.tools.apply_smote import apply_smote_with_group_mapping

__all__ = ['FeatureSet', 'Fold']

logger = logging.getLogger(__name__)

SCALER_MAPPING = {'standard': preprocessing.StandardScaler,
                  'minmax': preprocessing.MinMaxScaler,
                  'robust': preprocessing.RobustScaler, }


@dataclass
class FeatureSet:
    x: np.ndarray  # features (n_samples, n_features)
    y: np.ndarray  # labels (n_samples, )
    group: np.ndarray  # group mapping, (n_samples, ), maps the samples (nights) to the corresponding patient

    # feature mapping (n_features), maps the feature indices to the feature name
    feat_map: Optional[np.ndarray] = field(default=None)
    # feature ranking: maps each feature to its rank
    feat_rank: Optional[Dict] = field(default=None)
    # if features are processed (e.g. smote, scaling,), contains info about processing params
    process_params: Optional[Dict] = field(default=None)
    # if smote has been used, a flag indicating which samples are synthesized
    smote_mask: Optional[np.ndarray] = field(default=None)
    # if features were used for predictions, include predicted RBD probabilities (n_samples,)
    prob: Optional[np.ndarray] = field(default=None)
    # optional, a list of string labels, e.g. HC, RBD, PD, ..
    y_str: Optional[np.ndarray] = field(default=None)
    # dict-type flag to indicate association to Fold, default is None, i.e. no Fold instance associated
    from_fold: Optional[dict] = field(default=None)

    def __str__(self):
        return (f"FeatureSet(x.shape={self.x.shape},"
                f" probs={self.prob.shape if isinstance(self.prob, np.ndarray) else None},"
                f" process_params={self.process_params},"
                f" from_fold={self.from_fold})")

    def copy(self):
        return deepcopy(self)

    def select_features(self, feature_names: Union[list, np.ndarray]):
        """ Select a subset of features based on provided indices.
        Parameters:
            :param feature_names: (array-like) Str names of features to select.
        Returns:
            :return: (FeatureSet) A new FeatureSet instance with selected features. """
        assert self.feat_map is not None, f"'feat_map' attribute must be set."
        selected_indices = [i for i, name in enumerate(self.feat_map) if name in feature_names]

        new_x = self.x[:, selected_indices]
        new_feat_map = self.feat_map[selected_indices]
        return FeatureSet(x=new_x, y=self.y, group=self.group, feat_map=new_feat_map,
                          process_params=self.process_params, prob=self.prob)

    def select_samples(self, sample_indices):
        """ Select a subset of samples based on provided indices.
        Parameters:
            :param sample_indices: (array-like):Indices of samples to select.
        Returns:
            :return: (FeatureSet) A new FeatureSet instance with selected samples. """
        new_x = self.x[sample_indices, :]
        new_y = self.y[sample_indices]
        new_group = self.group[sample_indices]
        new_prob = self.prob[sample_indices] if self.prob is not None else None
        return FeatureSet(x=new_x, y=new_y, group=new_group, feat_map=self.feat_map, process_params=self.process_params,
                          prob=new_prob)

    def fit_transform(self, scaler: str, use_smote: bool, smote_seed: int, scaling_order: Optional[str] = None,
                      rank_kwargs: Optional[dict] = None):
        """ Fit and transform the processing pipeline for the FeatureSet. Should be used on training data only.
        Parameters:
            :param scaler: (str) Type of scaler to apply ('standard', 'minmax', 'robust', or 'none').
            :param use_smote: (bool) Whether to apply SMOTE to the training data.
            :param smote_seed: (int) Seed for SMOTE.
            :param scaling_order: (str)
            :param rank_kwargs (Optional[dict]). Specifies whether and from where the feature rankings should be
                fetched, or computed if the files does not exist yet.
        Returns:
            FeatureSet: The updated FeatureSet instance (self). """
        self.process_params = {'scaler': {'name': scaler}, 'SMOTE': None, 'ranking': None}

        if rank_kwargs:  # get feature ranking
            assert scaling_order in ['before_ranking', 'after_ranking'], \
                f"'scaling_order' must be 'before_ranking' or 'after_ranking', not '{scaling_order}'."
            if scaling_order == 'before_ranking':
                self._apply_scaler()
                self.feat_rank = self._get_feature_ranking(rank_kwargs)
            else:
                self.feat_rank = self._get_feature_ranking(rank_kwargs)
                self._apply_scaler()
        else:  # no feature ranking needed
            self._apply_scaler()

        if use_smote:  # apply smote
            self._apply_smote(smote_seed)

        return self

    def transform(self, process_params: Dict, ignore_smote: bool = True):
        """ Transform the FeatureSet using precomputed parameters, with optional control over SMOTE.
        Parameters:
            :param process_params: (dict): Parameters for the processing steps.
            :param ignore_smote: (bool): If True (default), SMOTE will not be applied, regardless of process_params.

        Returns:
            FeatureSet: The updated FeatureSet instance (self).
        """
        self.process_params = process_params

        # apply the scaler
        scaler_info = process_params.get('scaler', {})
        self._apply_scaler(scaler_info)

        # optionally apply smote
        smote_info = process_params.get('SMOTE', {})
        if not ignore_smote and smote_info.get('used', False):
            smote_seed = smote_info.get('seed', None)
            if smote_seed is None:
                raise ValueError("SMOTE seed is missing in process_params.")
            self._apply_smote(smote_seed)

        return self

    def merge(self, other: "FeatureSet") -> "FeatureSet":
        """Merge two FeatureSet objects by concatenating their samples.
        Assumes that both feature sets have the same feature mapping and processing parameters.
        Parameters:
            :param other: (FeatureSet) The other FeatureSet to merge with self.
        Returns:
            :return: (FeatureSet) A new FeatureSet with merged data. """
        # check that feature maps match
        if self.feat_map is not None or other.feat_map is not None:
            if not np.array_equal(self.feat_map, other.feat_map):
                raise ValueError("Feature mappings do not match between the two FeatureSets.")
        # merge the arrays
        merged_x = np.vstack((self.x, other.x))
        merged_y = np.concatenate((self.y, other.y))
        merged_group = np.concatenate((self.group, other.group))
        # optionally merge probabilities if both are present
        merged_prob = None
        if self.prob is not None and other.prob is not None:
            merged_prob = np.concatenate((self.prob, other.prob))

        return FeatureSet(
            x=merged_x, y=merged_y, group=merged_group, feat_map=self.feat_map, process_params=None, prob=merged_prob)

    def _apply_scaler(self, scaler_info: Optional[dict] = None):
        """ Applies scaling to the FeatureSet using the specified or precomputed scaler parameters.
        Parameters:
            :param scaler_info: (Optional[Dict]): The parameters for the scaler. If None, uses
            self.process_params['scaler'] (i.e. for fit_transform), else uses dict for .transform().
        Updates:
            self.x: Transformed features.
            self.process_params['scaler']: Updated scaler parameters. """
        if scaler_info is None:  # .fit_transform() case
            scaler_info = self.process_params['scaler']

        if scaler_info.get('name', '').lower() != 'none':
            scaler_class = SCALER_MAPPING.get(scaler_info.get('name').lower())
            if scaler_class is None:
                raise ValueError(f"Scaler type '{scaler_info.get('name')}' is not supported. "
                                 f"Choose from {list(SCALER_MAPPING.keys())} or 'none'.")

            scaler = scaler_class()
            if not scaler_info.get('fitted', False):  # scaler is not fitted
                self.x = scaler.fit_transform(self.x)
                scaler_info['fitted'] = True
                for attr in dir(scaler):  # dynamically store scaling parameters
                    if not attr.startswith('_') and hasattr(scaler, attr):
                        value = getattr(scaler, attr)
                        if isinstance(value, (list, np.ndarray, float, int)):
                            scaler_info[attr] = value
                logger.info(f" fitted and applied {scaler_info.get('name')} scaler.")

            else:  # apply fitted scaler
                for attr, value in scaler_info.items():  # set the scalers internal attributes to mimic a fitted state
                    if attr.endswith("_"):  # indicating fitted params
                        setattr(scaler, attr, np.array(value))
                self.x = scaler.transform(self.x)
                logger.info(f" applying pre-fitted {scaler_info.get('name')} scaler.")

        self.process_params['scaler'] = scaler_info

    def _apply_smote(self, seed: int):
        """ Applies SMOTE (Synthetic Minority Oversampling Technique) to the FeatureSet.
        Parameters:
            :param seed (int): Seed for SMOTE reproducibility.
        Updates:
            self.x: Augmented features.
            self.y: Augmented labels.
            self.group: Updated group mappings.
            self.smote_mask: Mask indicating synthetic samples.
            self.process_params['SMOTE']: SMOTE-related parameters."""
        # apply smote (logging handled internally)
        x_smote, y_smote, group_smote, smote_mask = apply_smote_with_group_mapping(self.x, self.y, self.group, seed)
        # update class attributes
        self.x, self.y, self.group, self.smote_mask = x_smote, y_smote, group_smote, smote_mask
        num_new_samples = x_smote.shape[0] - self.x.shape[0]
        self.process_params['SMOTE'] = {'used': True, 'num_new_samples': num_new_samples, 'seed': seed}

    def _get_feature_ranking(self, rank_kwargs: dict):
        """ Applies feature ranking to the FeatureSet based on the specified ranking method.
        Parameters:
            :param rank_kwargs (Dict): Arguments for the ranking method.
        Returns:
                Dict: Feature rankings (maps each feature to its rank)."""
        from aktiRBD.features import ranking
        # fetch or compute feature ranks
        return ranking.fetch_or_compute_feat_ranks(self, **rank_kwargs)


@dataclass
class Fold:
    name: str
    k: int
    feature_set: FeatureSet

    def __str__(self):
        return f"Fold({self.name}, k={self.k}, x={self.x.shape}, feature_set={self.feature_set})"

    def __post_init__(self):
        self.feature_set.from_fold = {'name': self.name, 'k': self.k}  # set the flag on the underlying FeatureSet:

    def __getattr__(self, attr):
        """Delegate methods and attributes to feature_set. Wrap returned FeatureSet
        objects in a Fold to preserve the interface."""
        fs = object.__getattribute__(self, "feature_set")  # avoid RecursionError
        feature_set_attr = getattr(fs, attr)
        if callable(feature_set_attr):
            def _delegated_method(*args, **kwargs):
                result = feature_set_attr(*args, **kwargs)
                if isinstance(result, FeatureSet):  # if delegated method returns FeatureSet, wrap it in a Fold.
                    if result is self.feature_set:
                        return self  # if same internal FeatureSet, return self
                    else:
                        return Fold(name=self.name, k=self.k, feature_set=result)
                return result

            return _delegated_method
        return feature_set_attr


if __name__ == '__main__':
    import numpy as np
    from aktiRBD.utils import setup_logging
    from aktiRBD.classifier.tools.feature_set import FeatureSet, Fold

    # Initialize logging
    setup_logging()

    # Set random seed for reproducibility
    np.random.seed(42)

    # Parameters for data generation
    n_features = 50
    n_class0 = 90  # Majority class
    n_class1 = 30  # Minority class (to allow SMOTE to generate synthetic samples)

    # Generate training data for class 0
    x_train_class0 = np.random.randn(n_class0, n_features) + 0  # Centered at 0
    y_train_class0 = np.zeros(n_class0, dtype=int)
    group_train_class0 = np.random.randint(1, 6, size=n_class0)  # Random group IDs between 1 and 5

    # Generate training data for class 1
    x_train_class1 = np.random.randn(n_class1, n_features) + 5  # Centered at 5
    y_train_class1 = np.ones(n_class1, dtype=int)
    group_train_class1 = np.random.randint(1, 6, size=n_class1)

    # Combine training data
    x_train = np.vstack((x_train_class0, x_train_class1))
    y_train = np.concatenate((y_train_class0, y_train_class1))
    group_train = np.concatenate((group_train_class0, group_train_class1))

    # Feature mapping
    feat_map = np.array([f'feature{i + 1}' for i in range(n_features)])

    # Generate test data
    n_test_samples = 6
    x_test = np.random.randn(n_test_samples, n_features) + 2  # Centered at (2,2,2)
    y_test = np.random.choice([0, 1], size=n_test_samples, p=[0.5, 0.5])  # Balanced classes in test set
    group_test = np.random.randint(1, 6, size=n_test_samples)

    # Create FeatureSet instances
    train_set = FeatureSet(x=x_train, y=y_train, group=group_train, feat_map=feat_map)
    test_set = FeatureSet(x=x_test, y=y_test, group=group_test, feat_map=feat_map)

    # Create Fold instances for training and testing
    train_fold = Fold(name="Train Fold", k=1, feature_set=train_set)
    test_fold = Fold(name="Test Fold", k=1, feature_set=test_set)

    # print(train_fold)
    # print(train_fold.copy())
    from pathlib import Path
    from aktiRBD.config import DataConfig

    _kwargs = {'scaler': 'robust', 'use_smote': True, 'smote_seed': 42, 'scaling_order': 'before_ranking',
               'rank_kwargs': {
                   'rank_path': Path('/Users/david/Desktop/test'), 'data_config': DataConfig, 'n_jobs': -1, }}
    train_processed = train_fold.copy().fit_transform(**_kwargs)
    print(type(train_processed))

    raise SystemExit
    for _scaler in ['standard', 'minmax', 'robust', 'none']:
        # Fit and transform the training set
        train_fold.fit_transform(
            scaler=_scaler, use_smote=True, smote_seed=42, rank_kwargs=None, scaling_order="before_ranking")

        # Transform the test set using the process_params from the training set
        # print(train_fold.feature_set.process_params)
        test_fold.transform(train_fold.feature_set.process_params)
        print(test_fold.copy())

        # Print process parameters for verification
        print(f"{_scaler} with SMOTE Process Params:")
        print(list(train_fold.feature_set.process_params.keys()))
        print(train_fold.feature_set.process_params)
        break
