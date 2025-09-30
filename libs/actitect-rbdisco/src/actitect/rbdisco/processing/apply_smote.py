import logging
from collections import defaultdict

import numpy as np
from imblearn.over_sampling import SMOTE

logger = logging.getLogger(__name__)

__all__ = ['apply_smote_with_group_mapping']


def apply_smote_with_group_mapping(x_train: np.ndarray, y_train: np.ndarray, id_map_train: np.ndarray,
                                   dataset_map_train: np.ndarray,  seed: int, assign_same_group_id: bool = True):
    """ Applies SMOTE and updates the id_map_train to match the resampled x_train.
    Parameters:
        :param x_train: Training features
        :param y_train: Training labels
        :param id_map_train: Mapping of samples to their group IDs
        :param dataset_map_train: Mapping of samples to their datasets.
        :param seed: random state of SMOTE instance
        :param assign_same_group_id: Boolean to indicate whether to assign the same group ID to synthetic samples
    Returns:
        :return x_train_resampled: Resampled training features
        :return y_train_resampled: Resampled training labels
        :return new_id_map_train: Updated mapping of samples to their group IDs"""
    smote = SMOTE(random_state=seed)
    x_train_resampled, y_train_resampled = smote.fit_resample(x_train, y_train)

    # calculate the number of new samples created
    num_new_samples = x_train_resampled.shape[0] - x_train.shape[0]
    synthetic_mask = np.array([False] * len(id_map_train) + [True] * num_new_samples)  # new samples padded to end

    new_id_map_train = np.copy(id_map_train)
    if dataset_map_train is not None:
        _check = defaultdict(set)
        for gid, dset in zip(id_map_train, dataset_map_train):
            _check[gid].add(dset)
        conflicting = {k: v for k, v in _check.items() if len(v) > 1}
        if conflicting:
            raise ValueError(f"Inconsistent dataset assignments found: {conflicting}")
        map_group_to_dataset = {gid: dsets.pop() for gid, dsets in _check.items()}
        new_dataset_map_train = np.copy(dataset_map_train)
    else:
        new_dataset_map_train, map_group_to_dataset = None, None

    if num_new_samples:
        logger.info(f"(SMOTE) Over-sampled {num_new_samples} samples of the minority class.")
        if assign_same_group_id:  # Assign new IDs based on IDs of nearest neighbors from the same class
            # this was needed to avoid data leakage in cv but is not needed anymore after refactor
            # (processing applied within folds...)
            for i in range(num_new_samples):
                synthetic_idx = x_train.shape[0] + i  # index of the synthetic sample

                # filter training data by the same class
                same_class_mask = y_train == y_train_resampled[synthetic_idx]
                x_train_same_class = x_train[same_class_mask]
                id_map_train_same_class = id_map_train[same_class_mask]

                if x_train_same_class.shape[0] > 0:  # perform nearest-neighbor search only within the same class
                    nearest_neighbors = smote.nn_k_.kneighbors(
                        x_train_resampled[synthetic_idx].reshape(1, -1), n_neighbors=x_train_same_class.shape[0],
                        return_distance=False)[0]

                    nearest_neighbor_idx = nearest_neighbors[0]  # Use the first nearest neighbor
                    nn_id = id_map_train_same_class[nearest_neighbor_idx]
                    new_id_map_train = np.append(new_id_map_train, nn_id)
                    if dataset_map_train is not None:
                        new_dataset_map_train = np.append(new_dataset_map_train, map_group_to_dataset[nn_id])

                else:
                    logger.warning(
                        f"No valid neighbor found for synthetic sample {synthetic_idx}. Skipping assignment.")
                    new_id_map_train = np.append(new_id_map_train, f'smote-{i}')
                    if dataset_map_train is not None:
                        new_dataset_map_train = np.append(new_dataset_map_train, f'smote-{i}')

        else:  # assign unique IDs to synthetic samples
            new_ids = [f'smote-{i}' for i in range(num_new_samples)]
            new_id_map_train = np.concatenate([id_map_train, new_ids])
            if dataset_map_train is not None:
                new_dataset_map_train = np.concatenate([dataset_map_train, new_ids])
    else:
        new_id_map_train = id_map_train
        logger.warning('No synthetic samples were added by SMOTE.')

    return x_train_resampled, y_train_resampled, new_id_map_train, new_dataset_map_train, synthetic_mask
