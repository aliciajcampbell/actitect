import logging

import numpy as np
import pandas as pd
from typing import Union

from aktiRBD.classifier.tools.feature_set import FeatureSet, Fold

logger = logging.getLogger(__name__)

__all__ = ['aggregate_night_predictions_to_patient_level', 'dataframe_agg_night_2_patient']


def aggregate_night_predictions_to_patient_level(data: Union[FeatureSet, Fold], y_pred: np.ndarray = None, kwargs=None):
    """ Aggregate night-level predictions to patient-level based on specified thresholds and voting strategy.
    Parameters:
        :param data: (FeatureSet or Fold) containing the night-level data, incl. probability scores.
        :param y_pred: (np.ndarray) containing the binary predictions, i.e. the thresholded probas.
        :param kwargs: (dict, optional) of aggregation parameters,
            defaults to {'mean_prob_threshold': 0.5, 'majority_vote_frac': 2}.
    Returns:
        :return: DataFrame containing aggregated patient-level predictions. """
    assert data.prob is not None, \
        f"'data' instance of type '{type(data)}' needs to have values assigned to '.prob' field."
    if not kwargs:
        kwargs = {'mean_prob_threshold': .5, 'majority_vote_frac': 2}
    if y_pred is None:
        y_pred = np.zeros_like(data.y)

    _, _idx, _night_counts = np.unique(data.group, return_inverse=True, return_counts=True)
    night_counts = np.cumsum(np.eye(_night_counts.size)[_idx], axis=0) - 1
    nights_valid = night_counts[np.arange(len(data.group)), _idx].astype('int')
    _agg_2_night_df = pd.DataFrame(
        {'id': data.group, 'night': nights_valid, 'ground_truth': data.y, 'prob': data.prob, 'pred': y_pred})
    return dataframe_agg_night_2_patient(_agg_2_night_df, **kwargs)


def dataframe_agg_night_2_patient(night_df: pd.DataFrame, mean_prob_threshold: float, majority_vote_frac: int):
    """ Perform aggregation of night-level data to patient-level metrics.
    Parameters:
        :param night_df: (pd.DataFrame) DataFrame containing night-level data
            with columns ['id', 'night', 'ground_truth', 'prob', 'pred'].
        :param mean_prob_threshold: (float) Threshold to determine positive predictions based on
            mean probability per night.
        :param majority_vote_frac: (int) Fraction to determine the majority vote threshold for aggregating predictions.
    Returns:
           :return: DataFrame containing aggregated patient-level predictions and metrics. """

    # assert that each group contains uniform labels
    assert all(night_df.groupby('id')['ground_truth'].nunique() == 1), \
        "Each 'id' group must contain uniform labels (only '1's or only '0's)."

    df_before_aggregation = pd.DataFrame(
        night_df.groupby(['id'])[['prob', 'pred', 'ground_truth', 'night']]
        .agg({'prob': ['mean', 'sum', 'max'], 'pred': ['mean', 'sum'], 'ground_truth': ['mean'], 'night': ['max']})
    ).reset_index()

    patient_df = pd.DataFrame()
    patient_df['id'] = df_before_aggregation.id
    patient_df['mean_prob_per_night'] = df_before_aggregation[('prob', 'mean')]  # mean RBD prob over all nights
    patient_df['sum_nights_prob'] = df_before_aggregation[('prob', 'sum')]  # sum of all probs (act. redundant)
    patient_df['mean_pred_per_night'] = df_before_aggregation[('pred', 'mean')]  # mean pred per night
    patient_df['max_prob_per_night'] = df_before_aggregation[('prob', 'max')]  # max. probability across all nights
    patient_df['sum_nights_pred'] = df_before_aggregation[('pred', 'sum')]  # sum of predicted nights
    patient_df['ground_truth'] = df_before_aggregation[('ground_truth', 'mean')]
    patient_df['n_total_nights'] = df_before_aggregation[('night', 'max')] + 1  # +1: night counter starts at 0
    del df_before_aggregation

    # threshold the mean probability per night
    patient_df['pred(mean_prob_per_night)'] = np.where(
        patient_df['mean_prob_per_night'] > mean_prob_threshold,
        1, 0
    )
    # majority voting # more than half of the nights are predicted positive?
    patient_df['pred(majority_vote)'] = (
            patient_df['sum_nights_pred'] >= (patient_df['n_total_nights'] / majority_vote_frac)
    ).astype(int)

    # ensemble: combine both. so basically
    _n_methods = 2  # change if I would add a method!
    ensemble_votes = patient_df[['pred(mean_prob_per_night)', 'pred(majority_vote)']].sum(axis=1)
    patient_df['pred(ensemble_major)'] = (ensemble_votes >= _n_methods - 1).astype(int)  # majority
    patient_df['pred(ensemble_all)'] = (ensemble_votes == _n_methods).astype(int)  # all
    # for n=2: ensemble_major is effectively 'OR' operator

    return patient_df


if __name__ == '__main__':
    from aktiRBD.classifier.tools import DataLoader
    from aktiRBD.classifier.tools.feature_set_old import process_feature_sets
    from aktiRBD.config import PipelineConfig
    from pathlib import Path

    from aktiRBD.utils import setup_logging

    setup_logging()

    _data_dir = Path('/data/processed')
    config = PipelineConfig.from_yaml(Path(
        '/aktiRBD/src/aktiRBD/config/pipeline.yaml'))
    _meta_file = Path('/Users/david/Desktop/py_projects/aktiRBD_private/data/raw/meta/metadata.csv')

    data_loader = DataLoader(_data_dir, _meta_file, **config.data.loader.dict())
    train_set, test_set, _ = data_loader.get_train_test_data(agg_level=config.data.agg_level)

    process_kwargs = config.data.processing.dict()
    process_kwargs.update({'smote_seed': 42})

    train_set, _ = process_feature_sets(train_set, None, rank_kwargs=None, **process_kwargs)
    print(np.unique(train_set.group, return_counts=True))

    train_set.prob = np.random.uniform(0, 1, train_set.y.shape)
    _per_patient_df_train = aggregate_night_predictions_to_patient_level(train_set)

    print(_per_patient_df_train.to_string())
