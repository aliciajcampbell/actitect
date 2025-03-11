from aktiRBD.classifier.tools.feature_set import FeatureSet, Fold

__all__ = ['cv_iterator']


def cv_iterator(cv_instance, data: FeatureSet, process_kwargs: dict = None, rank_kwargs: dict = None):
    """Iterator to split data into training and validation sets based on provided CV instance.
    Args:
        :param cv_instance: Grouped cross-validation instance with .split(groups) method (e.g., StratifiedGroupKFold).
        :param data: (FeatureSet) instance containing the data that should be iterated over.
        :param process_kwargs: (Dict)
        :param rank_kwargs: (Dict)
    Yields:
    tuple: (int, Fold, Fold) - Fold index and dataclass instances containing train and validation splits.
    """
    for k, (train_indices, val_indices) in enumerate(cv_instance.split(X=data.x, y=data.y, groups=data.group)):
        train_fold = Fold(name=f"train", k=k, feature_set=data.select_samples(train_indices))
        valid_fold = Fold(name=f"valid", k=k, feature_set=data.select_samples(val_indices))
        processed_train_fold = train_fold.copy().fit_transform(rank_kwargs=rank_kwargs, **process_kwargs)
        processed_valid_fold = valid_fold.copy().transform(processed_train_fold.process_params)
        assert isinstance(processed_train_fold.feat_rank, dict), \
            (f"'feat_rank' property of 'train_fold' inst. must be populated and of type dict,"
             f" not {processed_train_fold.feat_rank}")

        yield k, processed_train_fold, processed_valid_fold


if __name__ == '__main__':
    import numpy as np
    from sklearn.model_selection import StratifiedKFold
    from aktiRBD.classifier.tools import FeatureSet

    # Create random data
    n_samples = 100
    n_features = 33
    _x = np.random.rand(n_samples, n_features)
    _y = np.random.randint(0, 2, n_samples)
    _group = np.random.randint(0, 10, n_samples)
    feat_map = np.array([f"feature_{i}" for i in range(n_features)])

    # Create FeatureSet
    feature_set = FeatureSet(x=_x, y=_y, group=_group, feat_map=feat_map)

    # Create a CV instance
    _cv_instance = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Iterate through folds
    for _k, _train_fold, _valid_fold in cv_iterator(_cv_instance, feature_set):
        print(f"Fold {_k}:")
        print(_train_fold)
        print(_valid_fold)
        print(isinstance(_train_fold, Fold))
        print(isinstance(_train_fold, FeatureSet))
        print(type(_train_fold))
        break
