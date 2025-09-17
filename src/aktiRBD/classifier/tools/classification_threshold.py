import logging
from dataclasses import dataclass, asdict
from typing import Union, Optional

import numpy as np
from sklearn import metrics

from aktiRBD import utils
from aktiRBD.classifier.tools.aggregation import aggregate_night_predictions_to_patient_level
from aktiRBD.classifier.tools.feature_set import Fold, FeatureSet
from aktiRBD.utils.visualization import draw_roc_or_pr_curve

__all__ = ['get_operating_point', 'classify_with_threshold', 'eval_roc_and_pr_curves',
           'ClassThreshold', 'get_night_and_patient_threshold']

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
            _ = metrics.auc(fpr, tpr)
        elif curve_x is not None and curve_y is not None and thresholds is not None:
            fpr, tpr, thresholds = curve_x, curve_y, thresholds
            _ = metrics.auc(fpr, tpr)
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
            _ = metrics.auc(recall, precision)
        elif curve_x is not None and curve_y is not None and thresholds is not None:
            precision, recall, thresholds = curve_x, curve_y, thresholds
            _ = metrics.auc(recall, precision)
        else:
            raise ValueError("Either true_labels and predicted_probs, or x (precision), y (recall),"
                             " and thresholds must be provided for PR curve.")

        # avoid division by zero:
        f1_scores = np.divide(
            2 * (precision * recall), precision + recall,
            out=np.zeros_like(precision),
            where=(precision + recall) != 0
        )

        idx_opt_f1 = np.argmax(f1_scores)
        opt_threshold_f1 = _check_threshold_is_finite(thresholds[idx_opt_f1])

        distances = np.linalg.norm([precision - 1, recall - 1], axis=0)  # distance from (1,1)
        idx_opt_dist = np.argmin(distances)
        opt_threshold_dist = _check_threshold_is_finite(thresholds[idx_opt_dist])

        return {'dist': opt_threshold_dist, 'f1': opt_threshold_f1}

    else:
        raise ValueError("Invalid mode. Use 'roc' or 'pr'.")


def _parse_fixed_threshold(name: str) -> Optional[float]:
    """Return a float if `name` encodes a fixed threshold, else None.
       Accepts '0.42', 'fixed:0.42', 'fixed_0.42' (case-insensitive)."""
    s = str(name).strip()
    # direct float like "0.42"
    try:
        return float(s)
    except ValueError:
        pass
    # prefixed forms
    s_low = s.lower()
    for pref in ('fixed:', 'fixed_'):
        if s_low.startswith(pref):
            try:
                return float(s[len(pref):])
            except ValueError:
                return None
    return None


def _balanced_accuracy_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, thr: float) -> Optional[float]:
    """Balanced accuracy for a single subset at a fixed threshold; returns None if subset lacks both classes."""
    if len(y_true) == 0:
        return None
    uniq = np.unique(y_true)
    if uniq.size < 2:
        return None
    y_pred = (y_prob >= thr).astype(int)
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    # guard against zero division
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return 0.5 * (tpr + tnr)


def _candidate_thresholds(probs: np.ndarray, n: int = 401) -> np.ndarray:
    """hybrid approach from unique probs and grid (exact & robust)"""
    uniq = np.unique(probs)
    grid = np.linspace(0.0, 1.0, n)
    return np.unique(np.concatenate([grid, uniq, [0.0, 1.0]]))


def _macro_ba_threshold(
        y_true: np.ndarray, y_prob: np.ndarray, groups: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
    """Find threshold t that maximizes the (weighted) mean of per-group balanced accuracies.
    Groups are dataset labels (night- or patient-level).
    weights=None -> uniform over groups that have both classes.
    weights given -> re-normalized over valid groups."""
    assert y_true.shape == y_prob.shape == groups.shape, \
        f"mismatched shape {y_true.shape} {y_prob.shape} {groups.shape}"
    cand = _candidate_thresholds(y_prob)

    unique_groups = np.unique(groups)
    # pre-slice indices per group
    idx_by_g = {g: np.flatnonzero(groups == g) for g in unique_groups}

    # precompute group weights
    if weights is None:
        w_by_g = {g: 1.0 for g in unique_groups}
    else:
        assert weights.shape == groups.shape, f"mismatched shape {weights.shape} {groups.shape}"
        # derive weight per group as sum of sample weights in that group
        w_by_g = {g: float(np.sum(weights[idx_by_g[g]])) for g in unique_groups}

    best_thr, best_obj = .5, -np.inf

    for t in cand:
        bas, wts = [], []
        for g in unique_groups:
            idx = idx_by_g[g]
            ba = _balanced_accuracy_at_threshold(y_true[idx], y_prob[idx], t)
            if ba is None:
                continue  # skip groups without both classes
            bas.append(ba)
            wts.append(w_by_g[g])

        if not bas:
            continue

        wts = np.asarray(wts, dtype=float)
        wts = wts / wts.sum()  # normalize across valid groups
        obj = float(np.sum(wts * np.asarray(bas)))
        if obj > best_obj:
            best_obj = obj
            best_thr = t

    return float(best_thr)


def _single_method_threshold(y_true: np.ndarray, y_prob: np.ndarray, lvl: str, method: str) -> float:
    """Return one threshold for a given method using existing ROC/PR logic."""
    ths = eval_roc_and_pr_curves(y_true=y_true, y_prob=y_prob, lvl=lvl, mode='find_thresholds')
    lookup = {t.name.split(f"{lvl}_", 1)[-1]: t.value for t in ths}  # e.g. {'default':0.5,'roc_dist':..., ...}
    return float(lookup.get(method, lookup.get('default', 0.5)))


def get_night_and_patient_threshold(train: Union[Fold, FeatureSet], name: str):

    _fixed = _parse_fixed_threshold(name)
    if _fixed is not None:
        night = ClassThreshold(name=f"night_fixed({name})", value=float(_fixed))
        patient = ClassThreshold(name=f"patient_fixed({name})", value=float(_fixed))
        return night, patient

    base_methods = {'default', 'roc_dist', 'pr_dist', 'pr_f1'}
    macro_methods = {'macro_ba', 'macro_ba_weighted'}
    per_ds_prefixes = {'roc_dist_per_dataset_', 'pr_dist_per_dataset_', 'pr_f1_per_dataset_'}

    # Night-level
    if name in base_methods:
        night_ths = eval_roc_and_pr_curves(train.y, train.prob, lvl='night', mode='find_thresholds')
        night_val = {t.name.split('night_', 1)[-1]: t.value for t in night_ths}[name]
    elif name in macro_methods:
        # compute macro-BA night only
        ds = np.asarray(train.dataset)
        if name == 'macro_ba':
            night_val = _macro_ba_threshold(train.y, train.prob, ds, weights=None)
        else:
            _, counts = np.unique(ds, return_counts=True)
            w = np.array([dict(zip(np.unique(ds), counts))[d] for d in ds], float)
            night_val = _macro_ba_threshold(train.y, train.prob, ds, weights=w)
    elif any(name.startswith(p) for p in per_ds_prefixes):
        method = name.split('_per_dataset_')[0]  # e.g. 'roc_dist'
        comb = name.rsplit('_', 1)[-1]  # 'mean' or 'median'
        vals = []
        ds = np.asarray(train.dataset)
        for g in np.unique(ds):
            m = (ds == g)
            if np.unique(train.y[m]).size < 2:
                continue
            vals.append(_single_method_threshold(train.y[m], train.prob[m], lvl='night', method=method))
        night_val = float(np.mean(vals)) if comb == 'mean' else float(np.median(vals))
    else:
        raise ValueError(f"Unknown threshold name '{name}'")

    night = ClassThreshold(name=f"night_{name}", value=night_val)

    # Patient-level: mirror the same selection efficiently (no need to build all)
    df_pat = aggregate_night_predictions_to_patient_level(train)
    y_true_pat = df_pat['ground_truth'].to_numpy()
    y_prob_pat = df_pat['mean_prob_per_night'].to_numpy()

    # ===== Fast path for base methods: no dataset mapping needed =====
    if name in base_methods:
        pat_ths = eval_roc_and_pr_curves(y_true_pat, y_prob_pat, lvl='patient', mode='find_thresholds')
        patient_val = {t.name.split('patient_', 1)[-1]: t.value for t in pat_ths}[name]
        patient = ClassThreshold(name=f"patient_{name}", value=patient_val)
        return night, patient

    # map patient -> dataset only for macro/per-dataset methods
    if 'patient_id' in df_pat.columns:
        pid = df_pat['patient_id'].to_numpy()
        pid2ds = {}
        for p in np.unique(pid):
            m = (train.group == p)
            # handle single-dataset or missing dataset arrays gracefully
            if getattr(train, 'dataset', None) is None:
                u = np.array(['SINGLE'])
            else:
                u = np.unique(train.dataset[m])
            pid2ds[p] = u[0] if u.size else 'UNK'
        ds_pat = np.array([pid2ds.get(p, 'UNK') for p in pid])
    else:
        # fallback approximate grouping
        uniq_groups = np.unique(train.group)
        if getattr(train, 'dataset', None) is None:
            grp2ds = {g: 'SINGLE' for g in uniq_groups}
        else:
            grp2ds = {g: (np.unique(train.dataset[train.group == g])[0]
                          if np.unique(train.dataset[train.group == g]).size else 'UNK')
                      for g in uniq_groups}
        # recompute aligned arrays
        y_true_pat = np.array([int(np.round(np.mean(train.y[train.group == g]))) for g in uniq_groups])
        y_prob_pat = np.array([float(np.mean(train.prob[train.group == g])) for g in uniq_groups])
        ds_pat = np.array([grp2ds[g] for g in uniq_groups])

    if name in macro_methods:
        if name == 'macro_ba':
            patient_val = _macro_ba_threshold(y_true_pat, y_prob_pat, ds_pat, weights=None)
        else:
            _, counts = np.unique(ds_pat, return_counts=True)
            d2c = dict(zip(np.unique(ds_pat), counts))
            w = np.array([d2c[d] for d in ds_pat], float)
            patient_val = _macro_ba_threshold(y_true_pat, y_prob_pat, ds_pat, weights=w)
    else:  # per-dataset combine
        method = name.split('_per_dataset_')[0]
        comb = name.rsplit('_', 1)[-1]
        vals = []
        for g in np.unique(ds_pat):
            m = (ds_pat == g)
            if np.unique(y_true_pat[m]).size < 2:
                continue
            vals.append(_single_method_threshold(y_true_pat[m], y_prob_pat[m], lvl='patient', method=method))
        patient_val = float(np.mean(vals)) if comb == 'mean' else float(np.median(vals))

    patient = ClassThreshold(name=f"patient_{name}", value=patient_val)
    return night, patient


def _get_night_and_patient_thresholds(train: Union[Fold, FeatureSet]):
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
    """
    Calculate ROC and PR curves, determine optimal thresholds, and optionally plot and save the curves.

    Returns:
        list[ClassThreshold] (mode='find_thresholds') or None (mode='eval')
    """
    assert mode in ['eval', 'find_thresholds'], f"'mode' must be 'eval' or 'find_thresholds' not '{mode}'."
    opt_thresholds = [ClassThreshold(name=f'{lvl}_default', value=.5)] if mode == 'find_thresholds' else None

    # ROC
    _fpr, _tpr, _thresholds_roc = metrics.roc_curve(y_true, y_prob)
    _opt_threshold_roc = get_operating_point(curve_x=_fpr, curve_y=_tpr, thresholds=_thresholds_roc, mode='roc')
    if mode == 'eval':
        assert isinstance(eval_params, dict), "'eval_params' dict must be passed in 'eval' mode."
        required_keys = {'save_path', 'cv_mode'}
        missing_keys = required_keys - eval_params.keys()
        assert not missing_keys, f"'eval_params' is missing required keys: {missing_keys}"
        if eval_params.get('cv_mode'):
            assert 'n_interp_points_roc_pr' in list(eval_params.keys()), \
                f"if 'cv_mode', 'eval_params' requires additional argument 'n_interp_points_roc_pr' of type int."

        _roc_mode = {'curve': 'roc', 'lvl': lvl}
        _fig_roc = draw_roc_or_pr_curve(_fpr, _tpr, _thresholds_roc, _roc_mode)
        _fig_roc.savefig(eval_params.get('save_path').joinpath(f"{lvl}_roc_curve.png"), bbox_inches='tight', dpi=400)

        if eval_params.get('cv_mode'):
            _interp_fpr = np.linspace(0, 1, eval_params.get('n_interp_points_roc_pr'))
            _interp_tpr = np.interp(_interp_fpr, _fpr, _tpr)
            _interp_tpr[0] = 0.0
            utils.dump_to_json({
                'fpr': _interp_fpr, 'tpr': _interp_tpr,
                'auc': metrics.auc(_interp_fpr, _interp_tpr),
                'op_point': _opt_threshold_roc
            }, eval_params.get('save_path').joinpath(f"{lvl}_interp_roc_curve.json"))
    else:
        opt_thresholds.append(ClassThreshold(name=f'{lvl}_roc_dist', value=_opt_threshold_roc))

    # PR
    _pr, _rec, _thresholds_pr = metrics.precision_recall_curve(y_true, y_prob)
    _opt_threshold_pr = get_operating_point(curve_x=_pr, curve_y=_rec, thresholds=_thresholds_pr, mode='pr')
    if mode == 'eval':
        _n_hc, _n_rbd = np.unique(y_true, return_counts=True)[1]
        _pos_frac = _n_rbd / (_n_hc + _n_rbd)
        _pr_mode = {'curve': 'pr', 'lvl': lvl, 'pos_frac': _pos_frac}
        fig_pr = draw_roc_or_pr_curve(_rec, _pr, _thresholds_pr, _pr_mode)
        fig_pr.savefig(eval_params.get('save_path').joinpath(f"{lvl}_pr_curve.png"), bbox_inches='tight')

        if eval_params.get('cv_mode'):
            _interp_rec = np.linspace(0, 1, eval_params.get('n_interp_points_roc_pr'))
            _interp_pr = np.interp(_interp_rec, _rec[::-1], _pr[::-1])
            _interp_pr[0] = 1.0
            _interp_pr[-1] = _pos_frac
            utils.dump_to_json({
                'rec': _interp_rec, 'prec': _interp_pr,
                'f1_max': np.max(2 * (_interp_rec * _interp_pr) / (_interp_rec + _interp_pr)),
                'op_point': _opt_threshold_pr
            }, eval_params.get('save_path').joinpath(f"{lvl}_interp_pr_curve.json"))
    else:
        opt_thresholds.extend([ClassThreshold(name=f'{lvl}_pr_dist', value=_opt_threshold_pr['dist']),
                               ClassThreshold(name=f'{lvl}_pr_f1', value=_opt_threshold_pr['f1'])])

    return opt_thresholds
