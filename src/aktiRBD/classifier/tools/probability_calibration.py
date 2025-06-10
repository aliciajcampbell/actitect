import logging

import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.utils.validation import _check_sample_weight
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils.validation import check_is_fitted
from sklearn.model_selection import check_cv
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import indexable, _safe_indexing
from sklearn.utils.multiclass import check_classification_targets
from sklearn.base import clone

logger = logging.getLogger(__name__)
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'

__all__ = ['CustomCalibratedClassifierCV', 'draw_calibration_curve']


class CustomCalibratedClassifierCV(CalibratedClassifierCV):
    """ Custom subclass of CalibratedClassifierCV to enable grouped cv and early stopping. """

    def __init__(self, estimator=None, *, method: str = 'sigmoid', cv=None, n_jobs: int = None, ensemble: bool = True,
                 base_estimator: str = "deprecated", pass_fold_as_eval_set: bool = False):

        super().__init__(estimator=estimator, method=method, cv=cv, n_jobs=n_jobs, ensemble=ensemble,
                         base_estimator=base_estimator)

        self.calibrated_classifiers_, self.classes_ = None, None
        self.pass_fold_as_eval_set = pass_fold_as_eval_set  # added to enable early stopping.

    # noinspection PyPep8Namin
    def fit(self, X, y, sample_weight=None, groups=None, **fit_params):
        """ Same as CalibratedClassifierCV.fit except we pass 'groups' into the CV."""
        check_classification_targets(y)
        X, y = indexable(X, y)
        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X)

        estimator = self._get_estimator()
        self.calibrated_classifiers_ = []

        if self.cv == "prefit":  # if using prefit, skip CV entirely
            check_is_fitted(estimator, attributes=["classes_"])
            self.classes_ = estimator.classes_
            pred_method, method_name = self._get_prediction_method(estimator)
            n_classes = len(self.classes_)
            predictions = self._compute_predictions(pred_method, method_name, X, n_classes)

            calibrated_classifier = self._fit_calibrator(
                estimator, predictions, y, self.classes_, self.method, sample_weight)
            self.calibrated_classifiers_.append(calibrated_classifier)
        else:
            # Create CV splitter, explicitly passing 'groups'
            cv = check_cv(self.cv, y, classifier=True)
            splits = list(cv.split(X, y, groups=groups))

            label_encoder_ = LabelEncoder().fit(y)
            self.classes_ = label_encoder_.classes_
            n_classes = len(self.classes_)

            # Check if each class has enough samples for #folds
            if hasattr(cv, "n_splits"):
                n_folds = cv.n_splits
                if any((y == cl).sum() < n_folds for cl in self.classes_):
                    raise ValueError(f"Requesting {n_folds}-fold CV but < {n_folds} examples for at least one class.")

            if self.ensemble:
                from joblib import Parallel, delayed
                parallel = Parallel(n_jobs=self.n_jobs)
                self.calibrated_classifiers_ = parallel(
                    delayed(self._fit_classifier_calibrator_pair)(
                        clone(estimator), X, y, train=train, test=test, method=self.method, classes=self.classes_,
                        sample_weight=sample_weight, fit_params=fit_params) for train, test in splits)
            else:
                # Follow the original cross_val_predict logic for ensemble=False
                from functools import partial
                from sklearn.model_selection import cross_val_predict

                this_estimator = clone(estimator)
                pred_method, method_name = self._get_prediction_method(this_estimator)

                predictions = self._compute_predictions(
                    partial(cross_val_predict, estimator=this_estimator, X=X, y=y, cv=cv, method=method_name,
                            n_jobs=self.n_jobs, fit_params=fit_params),
                    method_name, X, n_classes)

                this_estimator.fit(X, y, **fit_params)
                calibrated_classifier = self._fit_calibrator(
                    this_estimator, predictions, y, self.classes_, self.method, sample_weight)
                self.calibrated_classifiers_.append(calibrated_classifier)

        # Copy attribute logic from base class
        first_clf = self.calibrated_classifiers_[0].estimator
        if hasattr(first_clf, "n_features_in_"):
            self.n_features_in_ = first_clf.n_features_in_
        if hasattr(first_clf, "feature_names_in_"):
            self.feature_names_in_ = first_clf.feature_names_in_
        return self

    def _fit_classifier_calibrator_pair(self, estimator, X, y, train, test, method, classes, sample_weight=None,
                                        fit_params=None):
        """ Overridden to optionally insert 'eval_set' for early stopping. """
        from sklearn.calibration import _fit_calibrator, _compute_predictions, _check_fit_params, _get_prediction_method

        X_train, y_train = _safe_indexing(X, train), _safe_indexing(y, train)
        X_test, y_test = _safe_indexing(X, test), _safe_indexing(y, test)

        fit_params_train = _check_fit_params(X, fit_params, train)
        if self.pass_fold_as_eval_set:
            fit_params_train["eval_set"] = [(X_test, y_test)]

        if sample_weight is not None:
            sw_train = _safe_indexing(sample_weight, train)
            fit_params_train["sample_weight"] = sw_train

        # debugging:
        # print("Fold train groups:", np.unique(_groups[train]))
        # print("Fold test groups:", np.unique(_groups[test]))

        estimator.fit(X_train, y_train, **fit_params_train)

        n_classes = len(classes)
        pred_method, method_name = _get_prediction_method(estimator)
        predictions = _compute_predictions(pred_method, method_name, X_test, n_classes)

        sw_test = _safe_indexing(sample_weight, test) if sample_weight is not None else None

        return _fit_calibrator(estimator, predictions, y_test, classes, method, sw_test)


def draw_calibration_curve(y_train, y_prob_train, hist_bin_width=.025):
    import numpy as np
    from sklearn.metrics import brier_score_loss

    n_samples = y_train.shape[0]
    # if n_samples < 100:
    #     logger.warning(f'with n={n_samples} samples, the calibration curve is likely not very reliable.')
    n_bins = max(int(np.log2(n_samples) + 1), 5)  # Sturges' rule

    frac_of_pos, mean_pred_vals = calibration_curve(y_train, y_prob_train, n_bins=n_bins, strategy='quantile')

    brier_score = brier_score_loss(y_train, y_prob_train)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharex=True)
    plt.subplots_adjust(top=.92, right=.92, left=.08, bottom=.12, wspace=.12)
    ax1.scatter(mean_pred_vals, frac_of_pos, marker='^', s=90, ec='b', c='c',
                label=f'model (brier = {brier_score:.3f})', zorder=30)
    ax1.plot(mean_pred_vals, frac_of_pos, ls='--', c='b', zorder=20, lw=1.4)
    ax1.plot([0, 1], [0, 1], ls='-', c='k', lw=1, alpha=.6, label=f'perfectly calibrated', zorder=10)
    ax1.set_xlabel('mean predicted probability', fontsize=12, labelpad=15)
    ax1.set_ylabel('fraction of positives', fontsize=12, labelpad=15)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.tick_params(axis='both', which='major', direction='in', length=6, width=2, top=True, right=True, labelsize=10)
    ax1.minorticks_on()
    ax1.tick_params(axis='both', which='minor', direction='in', length=4, width=1, top=True, right=True)
    ax1.set_title('calibration curve', fontweight='bold', fontsize=12, pad=10)
    ax1.fill_between([0, 1], 0, [0, 1], color='c', alpha=.12)
    ax1.text(.66, .33, 'OVER CONFIDENT', color='c', fontsize=11, rotation=45, fontweight='normal',
             va='center', ha='center')
    ax1.fill_between([0, 1], [0, 1], 1, color='m', alpha=.12)
    ax1.text(.33, .66, 'UNDER CONFIDENT', color='m', fontsize=11, rotation=45, fontweight='normal', alpha=.7,
             va='center', ha='center')
    ax1.legend(frameon=False, loc=(.05, .85), fontsize=11)

    ax2.hist(y_prob_train, bins=np.arange(0, 1 + hist_bin_width, hist_bin_width),
             color='lightblue', ec='b', alpha=.9)
    ax2.set_xlabel('predicted probability', fontsize=12, labelpad=15)
    ax2.set_ylabel(f'frequency per {hist_bin_width} bin', fontsize=12, labelpad=35, rotation=-90)
    ax2.yaxis.set_label_position('right')
    ax2.yaxis.tick_right()
    ax2.tick_params(axis='both', which='major', direction='in', length=6, width=2, top=True, left=True, labelsize=10)
    ax2.minorticks_on()
    ax2.tick_params(axis='both', which='minor', direction='in', length=4, width=1, top=True, left=True)
    ax2.set_title('probabilities', fontweight='bold', fontsize=12, pad=10)
    return fig, brier_score
