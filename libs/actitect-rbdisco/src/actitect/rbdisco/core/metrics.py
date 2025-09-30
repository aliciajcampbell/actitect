import logging

import numpy as np
import sklearn.metrics as sklm

from actitect.classifier.tools.classification_threshold import classify_with_threshold
from actitect.utils import compute_mean_std_ci

__all__ = ['calc_evaluation_metrics']

logger = logging.getLogger(__name__)


def calc_evaluation_metrics(
        y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray = None, threshold: float = None,
        *,
        bootstrap_ci: bool = False, n_boot: int = 2000, confidence_level: float = .95,
        stratified: bool = True, seed: int = 42):
    """ Calculate common evaluation metrics for a binary classification task.
    Parameters:
        :param y_true: (np.ndarray) The true class labels of shape (n_samples,).
        :param y_prob: (np.ndarray) The class probability scores of the positive, i.e. '1' class of shape (n_samples,).
        :param y_pred (np.ndarray, Optional) The binary class prediction. Required if 'threshold' not provided.
        :param threshold: (float, Optional) The threshold for the binary prediction. Required if 'y_pred' not provided.
        :param bootstrap_ci: (bool, optional) Whether to use bootstrap confidence intervals.
        :param n_boot: (int, Optional) Number of bootstrap samples to use.
        :param confidence_level: (float, Optional) The confidence level of the confidence intervals.
        :param stratified: (bool, optional) Whether to use stratified or unstratified.
        :param seed: (int, Optional) Seed for the random number generator.
    Returns:
        :return: (dict) containing the metrics and a mean+std value over all metrics.

        - **accuracy** (`float`): Proportion of correct predictions (both true positives and true negatives)
            out of all samples. *Note:* Accuracy can be misleading in cases of class imbalance.

        - **precision** (`float`): Proportion of true positive predictions among all positive predictions. Measures the
            model's ability to avoid false positives.

        - **recall** (`float`): Also known as sensitivity, it is the proportion of true positives correctly identified
            out of all actual positive instances. Measures the model's ability to capture all positive cases.

        - **f1** (`float`): The harmonic mean of precision and recall. Balances the trade-off between precision and
            recall, especially useful for imbalanced classes.

        - **average_precision** (`float`):  Area Under the Precision-Recall Curve (PR AUC). Represents the trade-off
            between precision and recall across different thresholds. Particularly useful when the positive class is
            rare or when focusing on positive instances.

        - **roc_auc** (`float`): Area Under the Receiver Operating Characteristic Curve (ROC AUC). Indicates the model's
            ability to distinguish between positive and negative classes across all thresholds.

        - **balanced_accuracy** (`float`): The arithmetic mean of sensitivity (recall) and specificity. Provides a
            balanced assessment of the model's performance across all classes, useful for imbalanced datasets.

        - **mean** (`float`): The mean of all the above metric values. Provides an overall average performance score.

        - **std** (`float`): The standard deviation of all the above metric values. Indicates the variability or
        dispersion of the metric scores. """
    if (threshold is not None) and (y_pred is not None):
        raise ValueError("provide either 'threshold' to compute 'y_pred' or provide 'y_pred' directly, not both.")
    elif (threshold is None) and (y_pred is None):
        raise ValueError("either 'threshold' or 'y_pred' must be provided.")

    if threshold is not None:
        y_pred = classify_with_threshold(y_prob, threshold)
    else:  # ensure 'y_pred' is binary
        unique_preds = np.unique(y_pred)
        if not np.all(np.isin(unique_preds, [0, 1])):
            raise ValueError("'y_pred' must contain only binary values 0 and 1.")

    metrics = {'accuracy': sklm.accuracy_score(y_true, y_pred),
               'precision': sklm.precision_score(y_true, y_pred),
               'recall': sklm.recall_score(y_true, y_pred),
               'f1': sklm.f1_score(y_true, y_pred),
               'average_precision': sklm.average_precision_score(y_true, y_prob),
               'roc_auc': sklm.roc_auc_score(y_true, y_prob),
               'balanced_accuracy': sklm.balanced_accuracy_score(y_true, y_pred)}
    # 'cohen_kappa': sklm.cohen_kappa_score(y_true, y_pred),
    # 'matthews_corr': sklm.matthews_corrcoef(y_true, y_pred)
    # takes into account tp, tn, fp, fn
    # 1: perfect prediction, 0: random, -1: total disagreement

    # add overall mean and std
    metrics.update({'mean': np.mean(list(metrics.values())), 'std': np.std(list(metrics.values()))})

    if not bootstrap_ci:
        return metrics

    # bootstrap to get stability estimate of each metric
    rng = np.random.default_rng(seed=seed)
    pos_idx, neg_idx = np.where(y_true == 1)[0], np.where(y_true == 0)[0]
    metric_names = [key for key in metrics.keys() if key not in ('mean', 'std')]
    boot_vals = {k: np.empty(n_boot) for k in metric_names}

    for b in range(n_boot):
        if stratified:
            res_pos = rng.choice(pos_idx, size=len(pos_idx), replace=True)
            res_neg = rng.choice(neg_idx, size=len(neg_idx), replace=True)
            idx = np.concatenate([res_pos, res_neg])
        else:
            idx = rng.choice(len(y_true), size=len(y_true), replace=True)

        yt, yp, yb = y_true[idx], y_prob[idx], y_pred[idx]

        try:
            boot_vals['accuracy'][b] = sklm.accuracy_score(yt, yb)
            boot_vals['precision'][b] = sklm.precision_score(yt, yb)
            boot_vals['recall'][b] = sklm.recall_score(yt, yb)
            boot_vals['f1'][b] = sklm.f1_score(yt, yb)
            boot_vals['average_precision'][b] = sklm.average_precision_score(yt, yp)
            boot_vals['roc_auc'][b] = sklm.roc_auc_score(yt, yp)
            boot_vals['balanced_accuracy'][b] = sklm.balanced_accuracy_score(yt, yb)
        except ValueError:
            # Handle cases like division by zero (e.g. precision when no positive predictions)
            for k in metric_names:
                boot_vals[k][b] = np.nan

    # Compute mean, std, and CI using t-distribution
    for k in metric_names:
        _vals = boot_vals[k][~np.isnan(boot_vals[k])]
        if len(_vals) <= 1:
            continue

        mean, std, ci_lo, ci_hi, margin = compute_mean_std_ci(_vals, confidence_level)

        # store original full-sample value from point estimate
        metrics[k + '_point'] = metrics[k]

        # overwrite main entry with bootstrap mean
        metrics[k] = mean

        # Store bootstrap-based statistics
        metrics[k + '_std'] = std
        metrics[k + '_ci'] = [ci_lo, ci_hi]
        metrics[k + '_margin'] = margin

    return dict(sorted(metrics.items()))
