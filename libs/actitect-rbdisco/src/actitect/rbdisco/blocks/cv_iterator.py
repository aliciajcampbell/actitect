import logging

import numpy as np
from sklearn.model_selection import BaseCrossValidator, StratifiedKFold, RepeatedStratifiedKFold, \
    StratifiedGroupKFold, StratifiedShuffleSplit

from ..core import FeatureSet, Fold

__all__ = ['cv_iterator']

logger = logging.getLogger(__name__)
_STRATIFIED_TYPES = StratifiedKFold, RepeatedStratifiedKFold, StratifiedGroupKFold, StratifiedShuffleSplit


def cv_iterator(
        cv_instance: BaseCrossValidator,
        data: FeatureSet,
        process_kwargs: dict = None,
        rank_kwargs: dict = None,
        stratify_by_dataset_if_pooled: bool = False, *,
        groups: np.ndarray = None,
      ):
    """ Iterator to split data into training and validation sets based on provided CV instance.
    Args:
        :param cv_instance: Grouped cross-validation instance with .split(groups) method (e.g., StratifiedGroupKFold).
        :param data: (FeatureSet) instance containing the data that should be iterated over.
        :param groups: 1-D array of same shape as data used for group-aware splitting. If None (default), uses the
            groups of the passed FeatureSet instance, i.e. data.groups.
        :param process_kwargs: (Dict) of kwargs passed to `FeatureSet.fit_transform()`.
        :param rank_kwargs: (Dict) of kwargs passed to feature ranking.
        :param stratify_by_dataset_if_pooled: (bool): If True, combine class and dataset labels for stratification if
            data is pooled.
    Yields:
        tuple: (int, Fold, Fold) - Fold index and dataclass instances containing train and validation splits."""
    is_stratified = isinstance(cv_instance, _STRATIFIED_TYPES)
    y_strat = data.get_strat_labels(stratify_by_dataset_if_pooled) if is_stratified else None
    groups_used = data.group if groups is None else groups
    for k, (train_indices, val_indices) in enumerate(cv_instance.split(X=data.x, y=y_strat, groups=groups_used)):
        train_fold = Fold(name=f"train", k=k, feature_set=data.select_samples(train_indices))
        valid_fold = Fold(name=f"valid", k=k, feature_set=data.select_samples(val_indices))
        if process_kwargs is not None:

            processed_train_fold = train_fold.copy().fit_transform(rank_kwargs=rank_kwargs, **process_kwargs)
            processed_valid_fold = valid_fold.copy().transform(processed_train_fold.process_params)
            if processed_train_fold.feat_rank is None:
                logger.debug("No feature ranking (likely fixed-features mode). Skipping feat_rank check.")
            else:
                assert isinstance(processed_train_fold.feat_rank, dict) and processed_train_fold.feat_rank, (
                    f"'feat_rank' must be a populated dict if feature ranking is used, got {type(processed_train_fold.feat_rank)}"
                )
        else:
            processed_train_fold, processed_valid_fold = train_fold, valid_fold
            logger.warning("No processing or feature ranking kwargs provided. Skipping feature processing.")

        yield k, processed_train_fold, processed_valid_fold


if __name__ == '__main__':
    import numpy as np
    from sklearn.model_selection import StratifiedGroupKFold
    from actitect.classifier.tools.feature_set import FeatureSet
    from actitect.utils import utils

    utils.setup_logging()

    # Create mock data
    n_samples = 100
    n_features = 10
    _x = np.random.rand(n_samples, n_features)
    _group = np.random.randint(0, 10, n_samples)

    # Simulate 1:3 class imbalance
    n_pos = n_samples * 3 // 4
    n_neg = n_samples - n_pos
    _y = np.array([0] * n_neg + [1] * n_pos)

    # Simulate 2:3 dataset imbalance
    n_a = n_samples * 2 // 5
    n_b = n_samples - n_a
    _dataset = np.array(["A"] * n_a + ["B"] * n_b)

    # Shuffle to avoid dataset/class correlation
    rng = np.random.default_rng(42)
    idx = rng.permutation(n_samples)
    _x, _y, _group, _dataset = _x[idx], _y[idx], _group[idx], _dataset[idx]

    _feat_map = np.array([f"feature_{i}" for i in range(n_features)])
    feature_set = FeatureSet(x=_x, y=_y, group=_group, feat_map=_feat_map, dataset=_dataset)

    # StratifiedGroupKFold supports both stratification and grouping
    _cv_instance = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)

    for _k, _train_fold, _valid_fold in cv_iterator(_cv_instance, feature_set, stratify_by_dataset_if_pooled=True):
        print(f"Fold {_k}")
        print(f"Train groups: {np.unique(_train_fold.group)}")
        print(f"Valid groups: {np.unique(_valid_fold.group)}")
        print(f"Train datasets: {np.unique(_train_fold.dataset)}")
        print(f"Valid datasets: {np.unique(_valid_fold.dataset)}")
        # Class distribution
        train_class_vals, train_class_counts = np.unique(_train_fold.y, return_counts=True)
        valid_class_vals, valid_class_counts = np.unique(_valid_fold.y, return_counts=True)

        print("Train class distribution:", dict(zip(train_class_vals, train_class_counts)))
        print("Train class %:",
              {k: f"{v / sum(train_class_counts):.1%}" for k, v in zip(train_class_vals, train_class_counts)})

        print("Valid class distribution:", dict(zip(valid_class_vals, valid_class_counts)))
        print("Valid class %:",
              {k: f"{v / sum(valid_class_counts):.1%}" for k, v in zip(valid_class_vals, valid_class_counts)})

        # Dataset distribution
        train_ds_vals, train_ds_counts = np.unique(_train_fold.dataset, return_counts=True)
        valid_ds_vals, valid_ds_counts = np.unique(_valid_fold.dataset, return_counts=True)

        print("Train dataset distribution:", dict(zip(train_ds_vals, train_ds_counts)))
        print("Train dataset %:",
              {k: f"{v / sum(train_ds_counts):.1%}" for k, v in zip(train_ds_vals, train_ds_counts)})

        print("Valid dataset distribution:", dict(zip(valid_ds_vals, valid_ds_counts)))
        print("Valid dataset %:",
              {k: f"{v / sum(valid_ds_counts):.1%}" for k, v in zip(valid_ds_vals, valid_ds_counts)})
        break
