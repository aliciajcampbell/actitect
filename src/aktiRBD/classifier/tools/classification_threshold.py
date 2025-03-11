import logging
from typing import Union, List
from dataclasses import dataclass, asdict

import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

from aktiRBD import utils
from aktiRBD.classifier.tools.feature_set import Fold, FeatureSet
from aktiRBD.classifier.tools.aggregation import aggregate_night_predictions_to_patient_level
from aktiRBD.utils.visualization import draw_roc_or_pr_curve

__all__ = ['get_operating_point', 'classify_with_threshold', 'eval_roc_and_pr_curves',
           'ClassThreshold', 'get_night_and_patient_threshold', 'get_night_and_patient_thresholds']

logger = logging.getLogger(__name__)


@dataclass
class ClassThreshold:
    name: str
    value: float

    def dict(self) -> dict:
        return asdict(self)


def classify_with_threshold(y_prob: np.ndarray, threshold: float):
    return (y_prob >= threshold).astype(int)


def get_operating_point(true_labels=None, predicted_probs=None, curve_x=None, curve_y=None, thresholds=None,
                        mode: str = 'roc', verbose: bool = True):
    def _check_threshold_is_finite(threshold: float):
        if not np.isfinite(threshold):
            if verbose:
                logger.warning(f"Non-finite threshold encountered: {threshold}. Setting to default threshold 0.5.")
            threshold = 0.5
        return threshold

    if mode == 'roc':
        if true_labels is not None and predicted_probs is not None:
            fpr, tpr, thresholds = metrics.roc_curve(true_labels, predicted_probs)
            _auc = metrics.auc(fpr, tpr)
        elif curve_x is not None and curve_y is not None and thresholds is not None:
            fpr, tpr, thresholds = curve_x, curve_y, thresholds
            _auc = metrics.auc(fpr, tpr)
        else:
            raise ValueError("Either true_labels and predicted_probs, or x (fpr), y (tpr),"
                             " and thresholds must be provided for ROC curve.")

        distance = np.linalg.norm([fpr - 0, tpr - 1], axis=0)  # distance from (0,1)
        idx_opt = np.argmin(distance)
        optimal_threshold = _check_threshold_is_finite(thresholds[idx_opt])

        return optimal_threshold

    elif mode == 'pr':
        if true_labels is not None and predicted_probs is not None:
            precision, recall, thresholds = metrics.precision_recall_curve(true_labels, predicted_probs)
            _auc = metrics.auc(recall, precision)
        elif curve_x is not None and curve_y is not None and thresholds is not None:
            precision, recall, thresholds = curve_x, curve_y, thresholds
            _auc = metrics.auc(recall, precision)
        else:
            raise ValueError("Either true_labels and predicted_probs, or x (precision), y (recall),"
                             " and thresholds must be provided for PR curve.")

        # avoid division by zero:
        f1_scores = np.divide(
            2 * (precision * recall), precision + recall,
            out=np.zeros_like(precision),  # set default to zero, i.e. when pr. = rec. = 0
            where=(precision + recall) != 0  # avoid division by zero
        )

        idx_opt_f1 = np.argmax(f1_scores)
        opt_threshold_f1 = _check_threshold_is_finite(thresholds[idx_opt_f1])

        distances = np.linalg.norm([precision - 1, recall - 1], axis=0)  # distance from (1,1)
        idx_opt_dist = np.argmin(distances)
        opt_threshold_dist = _check_threshold_is_finite(thresholds[idx_opt_dist])

        return {'dist': opt_threshold_dist, 'f1': opt_threshold_f1}

    else:
        raise ValueError("Invalid mode. Use 'roc' or 'pr'.")


def get_night_and_patient_threshold(train: Union[Fold, FeatureSet], name: str):
    """wrapper around 'get_night_and_patient_thresholds' to get only one specific thresholds."""
    _valid_names = ('default', 'roc_dist', 'pr_dist', 'pr_f1')
    assert name in _valid_names, f"'name' must be in {_valid_names}, not '{name}'."

    def _select_threshold(threshold_list: List[ClassThreshold], level_prefix: str, threshold_name: str):
        desired_name = f"{level_prefix}_{threshold_name}"
        for th in threshold_list:
            if th.name == desired_name:
                return th
        # Optionally: log a warning and return a default threshold.
        logger.warning(f"Threshold {desired_name} not found. Using default.")
        return threshold_list[0]  # fallback to first threshold

    night_thresholds, patient_thresholds = get_night_and_patient_thresholds(train)

    return (_select_threshold(night_thresholds, 'night', name),
            _select_threshold(patient_thresholds, 'patient', name))


def get_night_and_patient_thresholds(train: Union[Fold, FeatureSet]):
    """ Calculate thresholds for binary prediction from given probabilities. Returns a dict containing the night and
    patient level thresholds for the training data. The two levels of thresholds are independent of each other since
    only probabilities are used and not thresholded predictions. """
    assert train.prob is not None, f"'train' instance of type '{type(train)}' needs values assigned to '.prob' field."
    # calculate classification night-level thresholds based on roc/pr curves of training data
    night_thresholds = eval_roc_and_pr_curves(y_true=train.y, y_prob=train.prob, lvl='night', mode='find_thresholds')

    # aggregate to patient level and calculate patient thresholds (they are independent of respective night
    # threshold since only the non-thresholded probabilities, i.e. here the mean prob. per night is needed )
    _per_patient_df_train = aggregate_night_predictions_to_patient_level(train)

    _y_true_patient = _per_patient_df_train['ground_truth']
    _y_prob_patient = _per_patient_df_train['mean_prob_per_night']

    assert set(np.unique(_y_true_patient)) <= {0, 1}, "y_true must contain binary values (0 and 1)."
    assert np.all((_y_prob_patient >= 0) & (_y_prob_patient <= 1)), "y_prob must contain values between 0 and 1."

    patient_thresholds = eval_roc_and_pr_curves(
        y_true=_per_patient_df_train['ground_truth'],
        y_prob=_per_patient_df_train['mean_prob_per_night'],  # independent of night threshold
        lvl='patient', mode='find_thresholds')
    del _per_patient_df_train

    return night_thresholds, patient_thresholds


def eval_roc_and_pr_curves(y_true, y_prob, lvl: str, mode: str, eval_params: dict = None):
    """ Calculate ROC and PR curves, determine optimal thresholds, and optionally plot and save the curves.
    Parameters:
        :param y_true: (array-like) True binary labels.
        :param y_prob: (array-like) Predicted probabilities or scores.
        :param lvl: (str) Identifier for the evaluation level (e.g., 'night', 'patient').
        :param mode: (str) Operation mode, either 'eval' to plot/save curves or 'find_thresholds' to calc. thresholds.
        :param eval_params: (dict, Optional) specifies params for the evaluation, required for 'mode'='eval' and needs
            to have keys 'save_path', 'cv_mode' (bool) and  'n_interp_points_roc_pr' (int).

    Returns:
        :return: list[ClassThreshold] or None:
            - If mode is 'find_thresholds', returns a list of optimal thresholds. (to be fitted on train data only!)
            - If mode is 'eval', returns None."""
    assert mode in ['eval', 'find_thresholds'], f"'mode' must be 'eval' or 'find_thresholds' not '{mode}'."
    opt_thresholds = [ClassThreshold(name=f'{lvl}_default', value=.5)] if mode == 'find_thresholds' else None

    # roc-curve:
    _fpr, _tpr, _thresholds_roc = metrics.roc_curve(y_true, y_prob)
    _opt_threshold_roc = get_operating_point(curve_x=_fpr, curve_y=_tpr, thresholds=_thresholds_roc, mode='roc')
    if mode == 'eval':  # eval on roc curve (to use with validation data)
        assert isinstance(eval_params, dict), "'eval_params' dict must be passed in 'eval' mode."
        required_keys = {'save_path', 'cv_mode'}
        missing_keys = required_keys - eval_params.keys()
        assert not missing_keys, f"'eval_params' is missing required keys: {missing_keys}"
        if eval_params.get('cv_mode'):
            assert 'n_interp_points_roc_pr' in list(eval_params.keys()), \
                f" if 'cv_mode', 'eval_params' requires additional argument 'n_interp_points_roc_pr' of type int."

        _roc_mode = {'curve': 'roc', 'lvl': lvl}
        _fig_roc = draw_roc_or_pr_curve(_fpr, _tpr, _thresholds_roc, _roc_mode)
        _fig_roc.savefig(eval_params.get('save_path').joinpath(f"{lvl}_roc_curve_.png"), bbox_inches='tight', dpi=400)

        if eval_params.get('cv_mode'):  # for cv_mode, store the data to later aggregate an average
            _interp_fpr = np.linspace(0, 1, eval_params.get('n_interp_points_roc_pr'))
            _interp_tpr = np.interp(_interp_fpr, _fpr, _tpr)
            _interp_tpr[0] = 0.0
            utils.dump_to_json({
                'fpr': _interp_fpr, 'tpr': _interp_tpr,
                'auc': metrics.auc(_interp_fpr, _interp_tpr),
                'op_point': _opt_threshold_roc
            }, eval_params.get('save_path').joinpath(f"{lvl}_interp_roc_curve.json"))
    else:  # collect optimized thresholds (to use on training data only)
        opt_thresholds.append(ClassThreshold(name=f'{lvl}_roc_dist', value=_opt_threshold_roc))

    # pr-curve:
    _pr, _rec, _thresholds_pr = metrics.precision_recall_curve(y_true, y_prob)
    _opt_threshold_pr = get_operating_point(curve_x=_pr, curve_y=_rec, thresholds=_thresholds_pr, mode='pr')
    if mode == 'eval':  # eval on roc curve (to use with validation data)
        _n_hc, _n_rbd = np.unique(y_true, return_counts=True)[1]
        _pos_frac = _n_rbd / (_n_hc + _n_rbd)
        _pr_mode = {'curve': 'pr', 'lvl': lvl, 'pos_frac': _pos_frac}
        fig_pr = draw_roc_or_pr_curve(_rec, _pr, _thresholds_pr, _pr_mode)
        fig_pr.savefig(eval_params.get('save_path').joinpath(f"{lvl}_pr_curve.png"), bbox_inches='tight')

        if eval_params.get('cv_mode'):  # for cv_mode, store the data to later aggregate an average
            _interp_rec = np.linspace(0, 1, eval_params.get('n_interp_points_roc_pr'))
            _interp_pr = np.interp(_interp_rec, _rec[::-1], _pr[::-1])
            _interp_pr[0] = 1.0
            _n_hc, _n_rbd = np.unique(y_true, return_counts=True)[1]
            _interp_pr[-1] = _pos_frac
            utils.dump_to_json({
                'rec': _interp_rec, 'prec': _interp_pr,
                'f1_max': np.max(2 * (_interp_rec * _interp_pr) / (_interp_rec + _interp_pr)),
                'op_point': _opt_threshold_pr
            }, eval_params.get('save_path').joinpath(f"{lvl}_interp_pr_curve.json"))
    else:  # collect optimized thresholds (to use on training data only)
        opt_thresholds.extend([ClassThreshold(name=f'{lvl}_pr_dist', value=_opt_threshold_pr['dist']),
                               ClassThreshold(name=f'{lvl}_pr_f1', value=_opt_threshold_pr['f1'])])

    return opt_thresholds



