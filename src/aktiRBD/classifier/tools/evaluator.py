import gc
import logging
from collections import defaultdict
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics

from aktiRBD import utils
from aktiRBD.classifier.tools.aggregation import aggregate_night_predictions_to_patient_level
from aktiRBD.classifier.tools.classification_threshold import classify_with_threshold, eval_roc_and_pr_curves, \
    ClassThreshold
from aktiRBD.classifier.tools.feature_set import FeatureSet, Fold
from aktiRBD.classifier.tools.metrics import calc_evaluation_metrics
from aktiRBD.config import ExperimentConfig, NestedCVConfig

__all__ = ['Evaluator']

logger = logging.getLogger(__name__)


class Evaluator:

    def __init__(self, save_path: Path, experiment: ExperimentConfig, thresholds: dict,
                 cv_mode: bool, output_patient_csv: bool = False, cv_config: NestedCVConfig = None, *,
                 bootstrap_ci: bool = False):

        self.save_path = save_path
        self.experiment = experiment
        self.thresholds = thresholds  # numerical thresholds ({'night': <ClassThreshold>, 'patient': <ClassThreshold>})
        self.cv_mode = cv_mode
        self.bootstrap_ci = bootstrap_ci
        self.cv_config = cv_config
        self.output_patient_csv = False if self.cv_mode else output_patient_csv
        self.extra_diagnostic_metrics = bool(getattr(self.cv_config, 'extra_diagnostic_metrics', False))
        if cv_mode:
            assert cv_config, f"if 'cv_mode' is True, 'cv_config' must be passed as 'NestedCVConfig' instance."

    def __str__(self):
        return f"Evaluator(exp={self.experiment.name}, cv_mode={self.cv_mode})"

    def evaluate(self, train_data: Union[FeatureSet, Fold], valid_data: Union[FeatureSet, Fold],
                 generate_night_output: bool):

        assert valid_data.prob is not None, \
            f"'valid_data' instance of type '{type(valid_data)}' need to have values assigned to '.prob' attribute."
        if train_data is not None:
            assert train_data.prob is not None, \
                f"if 'train_data' is provided, the '{type(train_data)}' needs to have values assigned to '.prob'."

        if generate_night_output:  # save night level evaluation
            night_threshold = self.thresholds.get('night')
            save_path_night = utils.check_make_dir(self.save_path.joinpath('night'), True, verbose=False)
            scores_night = self._evaluate_night_level(
                train_data, valid_data, night_threshold, save_path_night)
            utils.dump_to_json(scores_night, save_path_night.joinpath('night_scores.json'))

        self._evaluate_patient_level(
            valid_data, self.thresholds.get('night'), self.thresholds.get('patient'), self.output_patient_csv)

        plt.close('all')
        gc.collect()

    def _evaluate_night_level(self, train_data: Union[FeatureSet, Fold], valid_data: Union[FeatureSet, Fold],
                              threshold: ClassThreshold, save_path: Path):
        y_pred_valid = classify_with_threshold(valid_data.prob, threshold.value)

        # calculate and collect classification evaluation metrics
        scores_night = {}
        if valid_data is not None:
            scores_night['valid'] = calc_evaluation_metrics(
                valid_data.y, y_prob=valid_data.prob, threshold=threshold.value)
            scores_night['valid']['cm'] = metrics.confusion_matrix(valid_data.y, y_pred_valid)
        if train_data is not None:  # mostly just an artifact from monitoring overfitting
            y_pred_train = classify_with_threshold(train_data.prob, threshold.value)
            scores_night['train'] = calc_evaluation_metrics(
                train_data.y, y_prob=train_data.prob, threshold=threshold.value)
            scores_night['train']['cm'] = metrics.confusion_matrix(train_data.y, y_pred_train)

        # draw a roc and pr curve (in cv_mode, save an interpolated curve for averaging)
        _eval_params = {'save_path': save_path, 'cv_mode': self.cv_mode}
        if self.cv_mode:
            _eval_params.update({'n_interp_points_roc_pr': self.cv_config.n_interp_points_roc_pr})
        _ = eval_roc_and_pr_curves(valid_data.y, valid_data.prob, lvl='night', mode='eval', eval_params=_eval_params)

        if not self.cv_mode:  # additionally, log an info json about misclassified nights per patient
            if valid_data is not None:
                # log info about N of misclassified nights for each patient
                utils.dump_to_json(data_dict=self._analyze_misclassified_nights_per_patient(
                    true=valid_data.y, pred=y_pred_valid, id_map=valid_data.group),
                    file_path=save_path.joinpath('misclassified_nights_per_patient.json'))

        return scores_night

    def _evaluate_patient_level(self, valid_data: Union[FeatureSet, Fold], night_threshold: ClassThreshold,
                                patient_threshold: ClassThreshold, output_csv: bool = True):

        # aggregate night level predictions to patient level
        y_pred_valid_night = classify_with_threshold(valid_data.prob, night_threshold.value)
        _agg_kwargs = {'mean_prob_threshold': patient_threshold.value, 'majority_vote_frac': 2}
        per_patient_df = aggregate_night_predictions_to_patient_level(
            valid_data, y_pred=y_pred_valid_night, kwargs=_agg_kwargs)

        if not self.cv_mode and output_csv:
            self._save_patient_level_predictions_csv(per_patient_df, valid_data)

        patient_agg = self.experiment.patient_aggregation  # e.g., "ensemble_major"

        _key = f"pred({patient_agg})"

        _patient_pred_scoring = calc_evaluation_metrics(
            per_patient_df['ground_truth'], y_prob=per_patient_df['mean_prob_per_night'], y_pred=per_patient_df[_key],
            bootstrap_ci=self.bootstrap_ci or not self.cv_mode)
        _patient_pred_scoring.update({
            'cm': metrics.confusion_matrix(per_patient_df['ground_truth'], per_patient_df[_key])})

        utils.dump_to_json(_patient_pred_scoring, self.save_path.joinpath('patient_scores.json'))

        # draw a roc and pr curve (in cv_mode, save an interpolated curve for averaging)
        _eval_params = {key: getattr(self, key) for key in ['save_path', 'cv_mode']}
        if self.cv_mode:
            _eval_params.update({'n_interp_points_roc_pr': self.cv_config.n_interp_points_roc_pr})
        _ = eval_roc_and_pr_curves(per_patient_df['ground_truth'], per_patient_df['mean_prob_per_night'],
                                   lvl='patient', mode='eval', eval_params=_eval_params)

        # if in CV/LODO and diagnostics are enabled, write fold-level diagnostics
        if self.cv_mode and bool(getattr(self.cv_config, "extra_diagnostic_metrics", False)):
            diag = self._compute_diagnostic_metrics(per_patient_df, thr=patient_threshold.value, agg_col=_key)
            utils.dump_to_json(diag, self.save_path.joinpath('patient_prob_diagnostic.json'))
            logger.info(f"Wrote patient diagnostics to {self.save_path.joinpath('patient_prob_summary.json')}")

    @staticmethod
    def _analyze_misclassified_nights_per_patient(true, pred, id_map):
        """ Analyze and count the number of misclassified nights per patient for
        different cases (train, test_default, test_roc_thresh).
        :param: true (np.ndarray): containing true labels on night-which_split.
        :param: pred (np.ndarray): containing predicted labels on night-which_split.
        :param: id_map (np.ndarray): maps the night-which_split instances to the higher patient-which_split.
        :returns: dict: Dictionary with the count of misclassified nights for each patient, sorted by  number of night.
        """
        misclassified = true != pred
        misclassified_counts = defaultdict(int)
        for patient_id in id_map[misclassified]:
            misclassified_counts[patient_id] += 1
        misclassified_counts = dict(sorted(misclassified_counts.items(), key=lambda item: item[1], reverse=True))
        misclassified_counts.update({'note': 'if SMOTE is used, n_nights > 7 is possible for train set.'})
        return misclassified_counts

    def _save_patient_level_predictions_csv(self, per_patient_df: pd.DataFrame, valid_data: Union[FeatureSet, Fold]):
        """ Saves a CSV file for patient-level predictions for a given combination of night and patient thresholds. """
        nightly_probs_df = pd.DataFrame({'id': valid_data.group, 'prob': valid_data.prob})
        nightly_probs_df = (nightly_probs_df.groupby('id')['prob'].apply(lambda x: ','.join(map(str, x.tolist())))
                            .reset_index().rename(columns={'prob': 'nightly_probs'}))

        merged_df = per_patient_df.merge(nightly_probs_df, on='id', how='left')
        agg_col = f"pred({self.experiment.patient_aggregation})"
        columns = ['id', 'ground_truth', 'mean_prob_per_night']
        if agg_col in merged_df.columns:
            columns.append(agg_col)
        columns.append('nightly_probs')

        # Save the CSV file using a simplified naming scheme based on the experiment name
        csv_path = self.save_path.joinpath(f"{self.experiment.name}_patient_predictions.csv")
        merged_df[columns].to_csv(csv_path, index=False)
        logger.info(f"Saved patient-level predictions CSV to {csv_path}")

    @staticmethod
    def _compute_diagnostic_metrics(per_patient_df: pd.DataFrame, *, thr: float, agg_col: str) -> dict:
        from sklearn.metrics import brier_score_loss, confusion_matrix
        import numpy as np
        y_true = per_patient_df["ground_truth"].to_numpy().astype(int)
        y_prob = per_patient_df["mean_prob_per_night"].to_numpy().astype(float)
        if agg_col in per_patient_df.columns:
            y_pred = per_patient_df[agg_col].to_numpy().astype(int)
        else:
            y_pred = (y_prob >= thr).astype(int)

        # confusion matrix (TN, FP, FN, TP)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

        # rates
        sens = tp / (tp + fn) if (tp + fn) else 0.0  # recall / TPR
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        ppv = tp / (tp + fp) if (tp + fp) else 0.0
        npv = tn / (tn + fn) if (tn + fn) else 0.0
        fpr = 1 - spec

        # simple calibration summary
        brier = brier_score_loss(y_true, y_prob)
        mean_p = float(y_prob.mean())
        prev = float(y_true.mean())
        pred_pos_frac = float((y_pred == 1).mean())

        # reliability bins (ECE-ish)
        n_bins = max(5, min(20, int(np.sqrt(len(y_prob)))))  # heuristic
        bins = np.linspace(0.0, 1.0, n_bins + 1)

        # prevent 1.0 from mapping to an out-of-range bin:
        idx = np.digitize(y_prob, bins, right=True)
        idx = np.clip(idx, 1, n_bins)

        bin_stats = []
        ece = 0.0
        for b in range(1, n_bins + 1):
            mask = (idx == b)
            n_b = int(mask.sum())
            if n_b == 0:
                bin_stats.append({"bin": b, "n": 0})
                continue
            p_hat = float(y_prob[mask].mean())
            acc_b = float(y_true[mask].mean())
            w = n_b / len(y_prob)
            ece += w * abs(acc_b - p_hat)
            bin_stats.append({"bin": b, "n": n_b, "mean_prob": p_hat, "empirical_pos": acc_b})

        return {
            "n_patients": int(len(y_true)),
            "threshold_used": float(thr),
            "prevalence": prev,
            "mean_predicted_probability": mean_p,
            "brier_score": float(brier),
            "predicted_positive_fraction": pred_pos_frac,
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "sensitivity": sens, "specificity": spec, "fpr": fpr, "ppv": ppv, "npv": npv,
            "ece": float(ece),
            "calibration_bins": bin_stats,
        }
