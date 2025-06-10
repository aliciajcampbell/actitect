import copy
import logging
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn import metrics, model_selection

from aktiRBD.classifier.tools.classification_threshold import get_operating_point, classify_with_threshold

__all__ = ['perform_stratified_group_cv', 'perform_loocv']
logger = logging.getLogger(__name__)


@dataclass
class CVResults:
    scoring: dict
    roc_curves: dict = None
    history: dict = None


def perform_stratified_group_cv(
        model,
        x_train: np.ndarray,
        y_train: np.ndarray,
        y_strat: np.ndarray,
        group: np.ndarray,
        use_early_stopping: bool,
        n_folds: int,
        shuffle: bool,
        n_jobs: int,
        random_seed_splitting: int = None,
        patient_aggregation_function=None,
        return_history: bool = False,
        verbose: bool = False,
        debug_print: bool = False,
):
    """
    TODO: clean up function, some artifacts from non-parallelized version

    Perform k-fold cross validation using StratifiedGroupKfold to
    i) group: avoid data-leakage by having nights of one patient in both, the train and valid set for a given fold.
    ii) stratify: keep approx. the same class balance for each fold.

    Parameters:
        :param model: A sklearn estimator that has a .fit() function.
        :param x_train: (np.ndarray) the training data of shape (n_samples, n_features).
        :param y_train: (np.ndarray) the training labels of shape (n_samples,).
        :param y_strat: (np.ndarray) the stratification labels of shape (n_samples,).
        :param group: (np.ndarray) of shape (n_samples,) to map each sample to a group (e.g. patient).
        :param use_early_stopping: (bool) whether to use early stopping or not.
        :param n_folds: (int) how many splits to perform per repeat. Determines the train/valid ratio.
        :param shuffle: (bool) whether the data should be shuffled before being split into folds.
        :param n_jobs: (int) number of CPUs to use for parallel processing.
        :param return_history: (Optional, bool) whether to log the results of each fold/repeat, e.g. for boxplot.
        :param patient_aggregation_function: (Optional, function) function to aggregate predictions to patient level.
        :param random_seed_splitting: (Optional, int) seed the splitter.
        :param verbose: (Optional, bool) whether to log info and results or not.
        :param debug_print: (Optional, bool) whether to print debug info.

    Returns:
        :return: CVResults
            Containing scoring and optionally roc curves and history.
    """
    if return_history:
        assert patient_aggregation_function is not None, \
            "'patient_aggregation_function' must be provided if 'return_history' is True"

    sgkf = model_selection.StratifiedGroupKFold(
        n_splits=n_folds,
        shuffle=shuffle,
        random_state=random_seed_splitting,  # if set, same splits for each repeat
    )

    n_train_split = int((1 - 1 / n_folds) * x_train.shape[0])
    n_val_split = int((1 / n_folds) * x_train.shape[0])
    if verbose:
        logger.info(
            f"cross-validation settings: k = {n_folds} "
            f"-> average train = {n_train_split} ({n_train_split / x_train.shape[0] * 100:.1f} %)"
            f" validation = {n_val_split} ({n_val_split / x_train.shape[0] * 100:.1f} %)"
        )

    _methods = ['default_thresh', 'roc_thresh']
    scoring_history_per_night = {}
    scoring_history_per_patient = {}
    misclassified_patients_per_fold = {}
    _scoring_history_placeholder = {
        'accuracy': np.zeros(n_folds),
        'balanced_accuracy': np.zeros(n_folds),
        'precision': np.zeros(n_folds),
        'recall': np.zeros(n_folds),
        'f1': np.zeros(n_folds),
        'auc': np.zeros(n_folds),
        'class_thresh': np.zeros(n_folds)}

    if return_history:
        _num_data_points = 100
        roc_tprs_night = np.zeros((n_folds, _num_data_points))
        roc_tprs_patient = roc_tprs_night.copy()
        mean_roc_fpr = np.linspace(0, 1, _num_data_points)
    else:
        _num_data_points = roc_tprs_night = roc_tprs_patient = mean_roc_fpr = None

    def _update_scoring_history(scoring_history, placeholder, key, fold, y_true, y_pred, fpr, tpr, threshold):
        """
        Update the scoring history dictionary with the provided values.

        Parameters:
        - scoring_history: dict, the scoring history to update
        - key: str, the name of the current method (key name within the dict)
        - fold: int, the current fold index
        - y_true: array-like, true labels
        - y_pred: array-like, predicted labels
        - fpr: array-like, false positive rates for ROC curve
        - tpr: array-like, true positive rates for ROC curve
        - threshold: float, classification threshold
        - placeholder: dict, the placeholder dictionary for initializing

        Returns:
        - scoring_history: dict, the updated scoring history
        """
        if key not in scoring_history:
            scoring_history[key] = copy.deepcopy(placeholder)

        scoring_history[key]['accuracy'][fold] = metrics.accuracy_score(y_true, y_pred)
        scoring_history[key]['balanced_accuracy'][fold] = metrics.balanced_accuracy_score(y_true, y_pred)
        scoring_history[key]['precision'][fold] = metrics.precision_score(y_true, y_pred, zero_division=0)
        scoring_history[key]['recall'][fold] = metrics.recall_score(y_true, y_pred)
        scoring_history[key]['f1'][fold] = metrics.f1_score(y_true, y_pred)
        scoring_history[key]['class_thresh'][fold] = threshold
        scoring_history[key]['auc'][fold] = metrics.auc(fpr, tpr)
        return scoring_history

    def _cross_val_fold(k, train_index, val_index, model, x_train, y_train, group, use_early_stopping,
                        patient_aggregation_function, return_history, debug_print, mean_roc_fpr, _methods,
                        _scoring_history_placeholder):
        local_scoring_history_per_night = {}
        local_scoring_history_per_patient = {method: copy.deepcopy(_scoring_history_placeholder) for method in _methods}
        local_misclassified_patients_per_fold = {}
        roc_tprs_night = None
        roc_tprs_patient = None

        x_train_fold, y_train_fold = x_train[train_index], y_train[train_index]
        x_valid_fold, y_valid_fold = x_train[val_index], y_train[val_index]
        group_train = group[train_index]
        group_valid = group[val_index]

        if debug_print:
            n_hc_train, n_rbd_train = np.unique(y_train_fold, return_counts=True)[1]
            train_total = n_rbd_train + n_hc_train
            n_hc_valid, n_rbd_valid = np.unique(y_valid_fold, return_counts=True)[1]
            valid_total = n_rbd_valid + n_hc_valid
            print('\n\n')
            print(f" fold: {k}")
            print(f" train: RBD={n_rbd_train} ({n_rbd_train / train_total * 100:.1f}%)"
                  f" HC={n_hc_train} ({n_hc_train / train_total * 100:.1f}%)")
            print(f"train ids: {np.unique(group_train)}")
            print(f" valid: RBD={n_rbd_valid} ({n_rbd_valid / valid_total * 100:.1f}%)"
                  f" HC={n_hc_valid} ({n_hc_valid / valid_total * 100:.1f}%)")
            print(f"valid ids: {np.unique(group_valid)}")

        if use_early_stopping:
            _eval_set = [(x_train_fold, y_train_fold), (x_valid_fold, y_valid_fold)]
            model.set_params(**{'early_stopping_rounds': 10, 'eval_metric': ['logloss']})
            model.fit(x_train_fold, y_train_fold, eval_set=_eval_set, verbose=False)
            model.set_params(**{'early_stopping_rounds': None, 'eval_metric': None})  # reset
        else:
            model.fit(x_train_fold, y_train_fold)

        y_prob_valid = model.predict_proba(x_valid_fold)[:, 1]
        _fpr, _tpr, _thresholds = metrics.roc_curve(y_valid_fold, y_prob_valid)

        # use train data to find optimized threshold
        _opt_threshold = get_operating_point(true_labels=y_train_fold,
                                             predicted_probs=model.predict_proba(x_train_fold)[:, 1],
                                             mode='roc', verbose=False)

        if return_history:
            interp_tpr = np.interp(mean_roc_fpr, _fpr, _tpr)
            interp_tpr[0] = 0.0
            roc_tprs_night = interp_tpr

        y_pred_valid_default = model.predict(x_valid_fold)
        y_pred_valid_roc_thresh = classify_with_threshold(y_prob_valid, _opt_threshold)

        for _mthd, y_pred_valid, _thresh in zip(_methods,
                                                [y_pred_valid_default, y_pred_valid_roc_thresh],
                                                [.5, _opt_threshold]):
            local_scoring_history_per_night = _update_scoring_history(local_scoring_history_per_night,
                                                                      _scoring_history_placeholder,
                                                                      key=_mthd, fold=k,
                                                                      y_true=y_valid_fold, y_pred=y_pred_valid,
                                                                      fpr=_fpr, tpr=_tpr, threshold=_thresh)
        if return_history:
            _, _idx, _night_counts = np.unique(group_valid, return_inverse=True, return_counts=True)
            night_counts = np.cumsum(np.eye(_night_counts.size)[_idx], axis=0) - 1
            nights_valid = night_counts[np.arange(len(group_valid)), _idx].astype('int')
            _agg_2_night_df_valid = pd.DataFrame({
                'id': group_valid,
                'night': nights_valid,
                'ground_truth': y_valid_fold,
                'prob': y_prob_valid,
                'pred': y_pred_valid_default,
            })

            per_patient_valid_default_threshold = patient_aggregation_function(_agg_2_night_df_valid, .5, 2)
            y_true_valid_patient = per_patient_valid_default_threshold['ground_truth']
            y_prob_valid_patient = per_patient_valid_default_threshold['mean_prob_per_night']
            _fpr_patient, _tpr_patient, _thresholds_patient = metrics.roc_curve(y_true_valid_patient,
                                                                                y_prob_valid_patient)

            roc_curve_thres_per_patient = get_operating_point(curve_x=_fpr_patient, curve_y=_tpr_patient,
                                                              thresholds=_thresholds_patient, verbose=False)

            interp_tpr_patient = np.interp(mean_roc_fpr, _fpr_patient, _tpr_patient)
            interp_tpr_patient[0] = 0.0
            roc_tprs_patient = interp_tpr_patient
            del per_patient_valid_default_threshold

            for _thresh_mthd, _threshold in zip(_methods, [.5, roc_curve_thres_per_patient]):
                _per_patient_valid = patient_aggregation_function(_agg_2_night_df_valid, _threshold, 2)
                _per_patient_true_valid = _per_patient_valid['ground_truth']

                for _pred_name in ('pred(mean_prob_per_night)', 'pred(majority_vote)',
                                   'pred(ensemble_major)', 'pred(ensemble_all)'):
                    _misclassified_patients = _per_patient_valid[
                        _per_patient_valid['ground_truth'] != _per_patient_valid[_pred_name]].id.values
                    local_misclassified_patients_per_fold.update({f" fold {k}": _misclassified_patients})

                    _per_patient_pred_valid = _per_patient_valid[_pred_name]
                    local_scoring_history_per_patient = _update_scoring_history(local_scoring_history_per_patient,
                                                                                _scoring_history_placeholder,
                                                                                key=f"{_thresh_mthd}_{_pred_name}",
                                                                                fold=k,
                                                                                y_true=_per_patient_true_valid,
                                                                                y_pred=_per_patient_pred_valid,
                                                                                fpr=_fpr_patient, tpr=_tpr_patient,
                                                                                threshold=_threshold)
        return (
            k, local_scoring_history_per_night, local_scoring_history_per_patient,
            local_misclassified_patients_per_fold,
            roc_tprs_night, roc_tprs_patient)

    with Parallel(n_jobs=n_jobs) as parallel_cv:
        results = parallel_cv(delayed(_cross_val_fold)(
            k, train_index, val_index, model, x_train, y_train, group, use_early_stopping, patient_aggregation_function,
            return_history, debug_print, mean_roc_fpr, _methods, _scoring_history_placeholder)
                              for k, (train_index, val_index)
                              in enumerate(sgkf.split(x_train, y_strat, group)))

    # Initialize the dictionaries for aggregation
    scoring_history_per_night = {method: copy.deepcopy(_scoring_history_placeholder) for method in _methods}
    scoring_history_per_patient = {method: copy.deepcopy(_scoring_history_placeholder) for method in _methods}
    misclassified_patients_per_fold = {}
    roc_tprs_night = np.zeros((n_folds, _num_data_points)) if return_history else None
    roc_tprs_patient = np.zeros((n_folds, _num_data_points)) if return_history else None

    for k, night_history, patient_history, misclassified, roc_night, roc_patient in results:
        for method in _methods:
            for metric in night_history[method]:
                scoring_history_per_night[method][metric][k] = night_history[method][metric][k]
            for metric in patient_history[method]:
                scoring_history_per_patient[method][metric][k] = patient_history[method][metric][k]
        misclassified_patients_per_fold.update(misclassified)
        if return_history:
            roc_tprs_night[k] = roc_night
            roc_tprs_patient[k] = roc_patient

    scoring = {'night': {}, 'patient': {}}
    for _name, _history in zip(['night', 'patient'], [scoring_history_per_night, scoring_history_per_patient]):
        for _method, _data in _history.items():
            if _method not in scoring[_name]:
                scoring[_name][_method] = {}
            for metric in _data.keys():
                _mean = np.mean(_data[metric])
                _std = np.std(_data[metric])
                scoring[_name][_method][metric] = {'mean': _mean, 'std': _std}

    if verbose:
        for method, scores in scoring['night'].items():
            logger.info(f'Cross-validation results for {method} (mean±std):')
            for metric, values in scores.items():
                logger.info(f" - {metric:17}: {values['mean']:.3f}±{values['std']:.4f}")

    if return_history:
        # count the total appearances of misclassified ids:
        id_counts = defaultdict(int)
        for folds in misclassified_patients_per_fold.values():
            for ids in folds:
                for id_ in ids:
                    id_counts[id_] += 1
        sorted_id_counts = dict(sorted(id_counts.items(), key=lambda item: item[1], reverse=True))
        misclassified_patients_per_fold.update({'summary': sorted_id_counts})
        return CVResults(scoring=scoring,
                         roc_curves={'mean_fpr': mean_roc_fpr,
                                     'tpr_night': roc_tprs_night,
                                     'tpr_patient': roc_tprs_patient},
                         history={'night': scoring_history_per_night,
                                  'patient': scoring_history_per_patient,
                                  'misclassified_patients': misclassified_patients_per_fold})
    else:
        return CVResults(scoring=scoring, roc_curves=None, history=None)


def perform_loocv(
        model,
        x_train: np.ndarray,
        y_train: np.ndarray,
        id_map_train: np.ndarray,
        use_early_stopping: bool,
        verbose: bool = True,
):
    train_ids = np.unique(id_map_train)
    y_true, y_pred, y_prob = [], [], []

    for i, _valid_id in enumerate(train_ids):

        # split the data: valid on all nights from one patient per fold
        _train_nights = np.where(id_map_train != _valid_id)[0]
        _valid_nights = np.where(id_map_train == _valid_id)[0]
        _x_train, _y_train = x_train[_train_nights], y_train[_train_nights]
        _x_valid, _y_valid = x_train[_valid_nights], y_train[_valid_nights]

        # fit and inference:
        if use_early_stopping:
            _eval_set = [(_x_train, _y_train), (_x_valid, _y_valid)]
            model.set_params(**{'early_stopping_rounds': 10, 'eval_metric': ['logloss']})
            model.fit(_x_train, _y_train, eval_set=_eval_set, verbose=False)
            model.set_params(**{'early_stopping_rounds': None, 'eval_metric': None})  # reset
        else:
            model.fit(_x_train, _y_train)
        _y_pred_valid = model.predict(_x_valid)
        _y_prob_valid = model.predict_proba(_x_valid)[:, 1]
        y_true.append(_y_valid)
        y_pred.append(_y_pred_valid)
        y_prob.append(_y_prob_valid)

        if verbose:
            logger.info(f"\n\nFold: {i} (Test ID: {_valid_id})")
            acc_train = metrics.accuracy_score(_y_valid, _y_pred_valid)
            bal_acc_train = metrics.balanced_accuracy_score(_y_valid, _y_pred_valid)
            for j, _true in enumerate(_y_valid):
                logger.info(f"True: {_true}, Pred: {_y_pred_valid[j]}")
            logger.info(f"Fold: {i} - Acc: {acc_train * 100:.3f}%, Bal Acc: {bal_acc_train * 100:.3f}%")

    y_true_valid = np.array([night for patient in y_true for night in patient])
    y_prob_valid = np.array([night for patient in y_prob for night in patient])
    y_pred_valid_default = np.array([night for patient in y_pred for night in patient])

    roc_threshold = get_roc_operating_point(true=y_true_valid, prob=y_prob_valid)
    y_pred_valid_roc_tresh = classify_with_threshold(y_prob_valid, roc_threshold)

    scoring_default, scoring_roc_thresh = {}, {}
    for y_pred_valid, score_dict, thresh in zip(
            [y_pred_valid_default, y_pred_valid_roc_tresh],
            [scoring_default, scoring_roc_thresh],
            [.5, roc_threshold]
    ):
        accuracy = metrics.accuracy_score(y_true_valid, y_pred_valid)
        balanced_accuracy = metrics.balanced_accuracy_score(y_true_valid, y_pred_valid)
        precision = metrics.precision_score(y_true_valid, y_pred_valid)
        recall = metrics.recall_score(y_true_valid, y_pred_valid)
        f1 = metrics.f1_score(y_true_valid, y_pred_valid)

        score_dict.update({
            'class_threshold': thresh,
            'accuracy': accuracy,
            'balanced_accuracy': balanced_accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
        })
        all_metrics = [accuracy, balanced_accuracy, precision, recall, f1]
        score_dict.update({'mean': np.mean(all_metrics), 'std': np.std(all_metrics)})

    return y_true_valid, y_prob_valid, y_pred_valid_default, y_pred_valid_roc_tresh, scoring_default, scoring_roc_thresh
