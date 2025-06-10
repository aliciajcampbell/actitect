import json
import logging
import math
import sys
import tracemalloc
import uuid
from abc import ABC, abstractmethod
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedGroupKFold, LeaveOneGroupOut
from skopt.plots import plot_convergence

from aktiRBD import utils
from aktiRBD.classifier.models import model_factory, ModelSetup
from aktiRBD.classifier.tools.classification_threshold import get_night_and_patient_threshold
from aktiRBD.classifier.tools.cv_iterator import cv_iterator
from aktiRBD.classifier.tools.evaluator import Evaluator
from aktiRBD.classifier.tools.feature_set import FeatureSet, Fold
from aktiRBD.classifier.tools.hp_optimizers import BayesianOptCV
from aktiRBD.classifier.tools.metrics import calc_evaluation_metrics
from aktiRBD.config import PipelineConfig
from aktiRBD.utils import visualization

logger = logging.getLogger(__name__)

__all__ = ['KFoldNestedCV', 'LODONestedCV']


class NestedCVBase(ABC):

    def __init__(self, config: PipelineConfig, save_path: Path, early_stopping_options: list = None):
        self.config = config
        self.save_path = save_path
        self.random_state = config.random_state
        self.n_jobs = config.nested_cv.n_jobs
        self.n_datasets, self.inner_seeds, self.n_repeats, self.outer_splits = None, None, None, None

        self.early_stopping_options = early_stopping_options if early_stopping_options is not None \
            else [True, False] if config.model.which == 'xgboost' else [False]

    def __str__(self) -> str:

        def __fmt(val):
            """Return val as str, but show “?” when None."""
            return "?" if val is None else val

        flavour = self.__class__.__name__
        return (
            f"{flavour}("
            f"datasets={__fmt(self.n_datasets)}, "
            f"repeats={__fmt(self.n_repeats)}, "
            f"outer_splits={__fmt(getattr(self, 'outer_splits', None))},"
            f"inner_splits={self.config.nested_cv.inner_cv.n_splits}, "
            f"n_jobs={self.n_jobs}, seed={self.random_state})"
        )

    @abstractmethod
    def _get_outer_splitter(self, seed: int):
        """Should return a parametrised sklearn splitter instance (that supports grouping)."""
        raise NotImplementedError('abstract method')

    @abstractmethod
    def _get_outer_groups(self, data: FeatureSet):
        """Array passed as `groups=` to cv_iterator for the outer loop."""
        raise NotImplementedError('abstract method')

    def fit(self, data: FeatureSet):

        # post-init setup (needs data)
        self.n_datasets = self._count_unique_datasets(data)
        self.n_repeats = 1 if isinstance(self, LODONestedCV) else self.config.nested_cv.n_repeats
        self.outer_splits = self.n_datasets if isinstance(self, LODONestedCV) \
            else self.config.nested_cv.outer_cv.n_splits
        self.inner_seeds = self._generate_random_seeds(self.random_state, self.n_repeats, self.outer_splits)
        if data.dataset is None:
            ds_summary = {"single": len(data.y)}
        else:
            labels, counts = np.unique(data.dataset, return_counts=True)
            ds_summary = {str(_l): int(_c) for _l, _c in zip(labels, counts)}
        seeds_dict = dict(outer=[self.random_state + r for r in range(self.n_repeats)], inner=self.inner_seeds)
        _manifest_kwargs = dict(
            save_dir=self.save_path, flavour=self.__class__.__name__.replace("NestedCV", "").lower(),
            repeats=self.n_repeats, outer_folds_expected=self.outer_splits,
            inner_folds=self.config.nested_cv.inner_cv.n_splits, seeds=seeds_dict, dataset_summary=ds_summary)

        with utils.custom_tqdm(total=self.n_repeats) as pbar, self._manifest_guard(**_manifest_kwargs):
            pbar.set_description(f"[PROGRESS]: {self.__class__.__name__}")
            _process_kwargs = self.config.data.processing.dict()
            _process_kwargs.update({'smote_seed': self.random_state})
            logger.info(f"({self.__class__.__name__}): using processing params {_process_kwargs}.")

            tracemalloc.start()
            for repeat in range(self.n_repeats):
                outer_seed = self.random_state + repeat
                outer_cv = self._get_outer_splitter(outer_seed)
                _rankings_root_dir = self.config.nested_cv.load_path_cv_feature_rankings.joinpath(
                    f"seed_{outer_seed}_{sys.platform}")
                _rank_kwargs = {'root_dir': _rankings_root_dir, 'data_config': self.config.data, 'n_jobs': self.n_jobs}
                save_path_repeat = utils.check_make_dir(self.save_path.joinpath(
                    f"repeats/repeat{repeat}_seed{outer_seed}"), True)

                tasks = [delayed(self._perform_outer_fold)(
                    k_outer, train_outer, valid_outer, save_path_repeat, outer_seed=outer_seed
                ) for k_outer, train_outer, valid_outer in cv_iterator(
                    outer_cv, data, _process_kwargs, _rank_kwargs,
                    self.config.nested_cv.stratify_by_dataset_if_pooled, groups=self._get_outer_groups(data))]

                with Parallel(n_jobs=self.n_jobs) as parallel:
                    parallel(tasks)

                pbar.update(1)
                current, peak = tracemalloc.get_traced_memory()
                logger.info(f"repeat {repeat + 1}/{self.n_repeats}:"
                            f" current memory usage: {current / 1024 / 1024:.2f} MB;"
                            f" peak: {peak / 1024 / 1024:.2f} MB")

                tracemalloc.clear_traces()

            tracemalloc.stop()

    def _perform_outer_fold(self, k_outer, train_outer, valid_outer, save_path_repeat, outer_seed):
        with self._handle_parallel_logging():

            save_path_fold = utils.check_make_dir(save_path_repeat.joinpath(f'outer_fold_{k_outer}'), True)
            inner_seed = self.inner_seeds[outer_seed][k_outer]
            self._set_random_seed(inner_seed)
            utils.dump_to_json({'seed': outer_seed, 'fold': k_outer,
                                'train': np.unique(train_outer.group), 'valid': np.unique(valid_outer.group)},
                               save_path_fold.joinpath(f"groups.json"))

            # set up the model:
            model_setup = self._initialize_model(self.config.model.which, train_outer, inner_seed)

            # fit and eval non-optimized model:
            default_model = model_setup.model().set_params(**model_setup.default_params)
            _default_scores = self._fit_and_eval_model(default_model, train_outer, valid_outer)
            logger.info(f"default model outer fold {k_outer}: {utils.fmt_dict(_default_scores['valid'])}")
            utils.dump_to_json(_default_scores, save_path_fold.joinpath(f"default_model_scores.json"))
            del _default_scores

            # perform BayesOpt using Inner-CV:
            fit_params = self.config.model.bayes_params.dict().copy()
            fit_params.update({'cv_params': {
                'group': train_outer.group, 'n_folds': self.config.nested_cv.inner_cv.n_splits,
                'shuffle': self.config.nested_cv.inner_cv.shuffle, 'verbose': False},
                'n_jobs': 1 if self.n_jobs != 1 else -1})
            opt_params, best_score = self._perform_inner_cv_bayes_opt(
                outer_fold=train_outer, model_setup=model_setup, rank_map=train_outer.feat_rank,
                save_path=save_path_fold, fit_params=fit_params, inner_cv_seed=inner_seed,
                stratify_by_dataset=self.config.nested_cv.stratify_by_dataset_if_pooled)

            best_params = model_setup.bayesian_fixed_params.copy()
            best_params.update(opt_params)
            _top_k = best_params.pop('top_k_feats')

            _top_k_feat_names = {
                name: val['rank'] for name, val in train_outer.feat_rank.items() if val['rank'] <= _top_k}
            _top_k_feat_names = dict(sorted(_top_k_feat_names.items(), key=lambda item: item[1]))
            utils.dump_to_json(_top_k_feat_names, save_path_fold.joinpath(f"top_k_feat_names.json"))

            # choose only selected features
            train_outer_top_k = train_outer.select_features(list(_top_k_feat_names.keys()))
            valid_outer_top_k = valid_outer.select_features(list(_top_k_feat_names.keys()))

            # fit and eval optimized model:
            for early_stopping in self.early_stopping_options:

                opt_model = model_setup.model().set_params(**best_params)
                logger.info(f"Bayesian opt: setting model params to {utils.fmt_dict(opt_params, 3)}.")
                _eval_set = [(train_outer_top_k.x, train_outer_top_k.y), (valid_outer_top_k.x, valid_outer_top_k.y)]

                if early_stopping:
                    opt_model.set_params(**self.config.model.early_stopping.dict())
                    _suffix = 'early_stopping'
                else:
                    opt_model.set_params(**{'early_stopping_rounds': None, 'eval_metric': None})  # reset
                    _suffix = 'no_early_stopping'

                opt_model.fit(train_outer_top_k.x, train_outer_top_k.y, eval_set=_eval_set, verbose=False)
                opt_model.set_params(**{'early_stopping_rounds': None, 'eval_metric': None})  # reset
                save_path_early_stopping = utils.check_make_dir(save_path_fold.joinpath(_suffix), True)

                # predict probabilities and calculate thresholds
                train_outer_top_k.prob = opt_model.predict_proba(train_outer_top_k.x)[:, 1]
                valid_outer_top_k.prob = opt_model.predict_proba(valid_outer_top_k.x)[:, 1]

                fold_night_threshold, fold_patient_threshold = get_night_and_patient_threshold(
                    train_outer_top_k, self.config.nested_cv.default_experiment.threshold)
                thresholds = {'night': fold_night_threshold, 'patient': fold_patient_threshold}

                evaluator = Evaluator(
                    save_path_early_stopping, self.config.nested_cv.default_experiment, thresholds,
                    cv_config=self.config.nested_cv, cv_mode=True)
                evaluator.evaluate(
                    train_outer_top_k, valid_outer_top_k, generate_night_output=self.config.nested_cv.log_night_eval)

                del opt_model

            del model_setup

    @staticmethod
    def _perform_inner_cv_bayes_opt(outer_fold: Fold, model_setup: ModelSetup, rank_map: dict, fit_params: dict,
                                    save_path: Path, inner_cv_seed: int, stratify_by_dataset: bool,
                                    plot_search_history: bool = True):
        # run Bayesian Optimization on inner-cv:
        bayesian_opt = BayesianOptCV(model_setup=model_setup, feat_rank_map=rank_map, seed=inner_cv_seed)
        y_strat = outer_fold.get_strat_labels(stratify_by_dataset)
        results = bayesian_opt.fit(outer_fold.x, outer_fold.y, y_strat, **fit_params)

        opt_params = {dim.name: val for dim, val in zip(model_setup.bayesian_param_space, results.x)}
        logger.info(f"best score: {-results.fun:.3f} with params {utils.fmt_dict(opt_params, 3)}")

        # saving and plotting:
        save_path_bay_opt = utils.check_make_dir(save_path.joinpath(f'bay_opt'), use_existing=True)
        _best_params = {'best_params': opt_params, 'best score': -results.fun}
        for filename, _dict in zip(
                ['best_params', 'fixed_params', 'param_space'],
                [_best_params, model_setup.bayesian_fixed_params, model_setup.bayesian_param_space]
        ):
            utils.dump_to_json(_dict, save_path_bay_opt.joinpath(f"{filename}.json"))

        if plot_search_history:  # plot hp search
            plt.close('all')
            _plotting_funcs = {
                'convergence': (plot_convergence, {}),
                # 'evaluations': (plot_evaluations, {'cmap': 'YlGnBu_r'}),
                # 'objective': (plot_objective, {'sample_source': 'result', 'cmap': 'YlGnBu'})
            }
            for _name, (func, kwargs) in _plotting_funcs.items():
                for _ext in ['png']:  # , 'pdf'):
                    func(results, **kwargs).get_figure() \
                        .savefig(save_path_bay_opt.joinpath(f'{_name}.{_ext}'), bbox_inches='tight')
                plt.close('all')

        return opt_params, -results.fun

    @staticmethod
    def _initialize_model(model_name: str, fold: Fold, random_state: int):
        _n_hc_train_outer, _n_rbd_train_outer = np.unique(fold.y, return_counts=True)[1]
        _cls_balance_outer = _n_hc_train_outer / _n_rbd_train_outer
        return model_factory(model_name, cls_balance=_cls_balance_outer, seed=random_state)

    @staticmethod
    def _fit_and_eval_model(model, fold_train: Fold, fold_valid: Fold, threshold: float = .5):
        model.fit(fold_train.x, fold_train.y)
        return {'train': calc_evaluation_metrics(
            fold_train.y, y_prob=model.predict_proba(fold_train.x)[:, 1], threshold=threshold),
            'valid': calc_evaluation_metrics(
                fold_valid.y, y_prob=model.predict_proba(fold_valid.x)[:, 1], threshold=threshold)}

    def eval(self):
        fold_dirs = sorted([_dir for _dir in self.save_path.glob('**/outer_fold*') if _dir.is_dir()])

        _expected_outer = self.n_datasets if isinstance(self, LODONestedCV) else self.config.nested_cv.outer_cv.n_splits
        assert len(fold_dirs) == self.n_repeats * _expected_outer, \
            (f"incomplete outer cv detected, found {len(fold_dirs)} folds at {self.save_path.stem} "
             f"but got n_splits={_expected_outer}.")

        # HP* search summary
        hp_summary, top_k_summary = self._make_hp_summary(fold_dirs)
        hp_summary.to_csv(self.save_path.joinpath('hp_summary.csv'))
        top_k_summary.to_csv(self.save_path.joinpath('top_k_summary.csv'))

        # collect the scores anr roc/pr curves per fold:
        scores, curves, cms = self._collect_scoring_history(fold_dirs, self.early_stopping_options)

        # boxplot of the scores:
        bp_night, bp_patient = self._draw_history_boxplots(scores, self.config.nested_cv.log_night_eval)
        for _lvl, _fig_dict in zip(['night', 'patient'], [bp_night, bp_patient]):
            if _fig_dict:  # non-empty dict
                __save_path = utils.check_make_dir(self.save_path.joinpath(f'boxplots/{_lvl}'), True)
                for _name, _fig in _fig_dict.items():
                    _fig.savefig(__save_path.joinpath(f'{_name}.png'), dpi=400, bbox_inches='tight')

        # mean roc/pr curve plots:
        curve_figs_night, curve_figs_patient = self._draw_mean_curves(curves, self.config.nested_cv.log_night_eval)
        for _lvl, _fig_dict in zip(['night', 'patient'], [curve_figs_night, curve_figs_patient]):
            if _fig_dict:  # non-empty dict
                _save_path_curves = utils.check_make_dir(self.save_path.joinpath(f'curves/{_lvl}'), True)
                for _name, _fig in _fig_dict.items():
                    _fig.savefig(_save_path_curves.joinpath(f'{_name}.png'), dpi=400, bbox_inches='tight')

    @staticmethod
    def _generate_random_seeds(initial_seed, n_repeats, n_splits_outer):
        offset = n_repeats * n_splits_outer if initial_seed == 0 else 0
        factor = max(10, 10 ** (math.ceil(math.log10(n_repeats * n_splits_outer)) - 1))
        outer_seeds, inner_seeds = [initial_seed + i for i in range(n_repeats)], {}
        for i, outer_seed in enumerate(outer_seeds):
            inner_seeds[outer_seed] = [(offset + (outer_seed * factor) + j) for j in range(n_splits_outer)]
        all_seeds = outer_seeds + [seed for inner_list in inner_seeds.values() for seed in inner_list]
        assert len(all_seeds) == len(np.unique(all_seeds)), \
            "found non-unique seeds in 'inner_seeds'. Fix it to ensure reproducibility."
        return inner_seeds

    @staticmethod
    def _set_random_seed(seed):
        np.random.seed(seed)

    @staticmethod
    def _make_hp_summary(fold_dirs: list[Path]):
        opt_params, top_k_feats = pd.DataFrame(), pd.DataFrame()

        for k, _fold_dir in enumerate(fold_dirs):
            _opt_params = utils.read_from_json(_fold_dir.joinpath('bay_opt/best_params.json'))['best_params']
            opt_params = pd.concat([opt_params, pd.DataFrame([_opt_params]).T], axis=1)
            _top_k_feats = utils.read_from_json(_fold_dir.joinpath('top_k_feat_names.json'))
            top_k_feats = pd.concat([top_k_feats, pd.DataFrame(index=_top_k_feats.keys(),
                                                               data={f'fold_{k + 1}': 1})], axis=1, sort=False)

        top_k_feats['sum'] = top_k_feats.sum(axis=1)
        top_k_feats = top_k_feats.sort_values(by='sum', ascending=False).fillna(0)

        opt_params.columns = [f'fold_{i + 1}' for i in range(len(fold_dirs))]
        opt_params['mean'] = opt_params.mean(axis=1)
        opt_params['std'] = opt_params.std(axis=1)
        opt_params = opt_params.round(2)
        return opt_params, top_k_feats

    @staticmethod
    def _collect_scoring_history(fold_dirs: list[Path], early_stop_options: list) -> (dict, dict, dict):
        """
            Collects scoring history across folds for the new Evaluator output.
            For each fold and for each early-stopping option (i.e. "early_stopping" and "no_early_stopping"),
            it reads the following files (if available):
              - default_model_scores.json (at the root of each fold)
              - night_scores.json from the fixed subfolder (e.g. <fold_dir>/<es_key>/night/night_scores.json)
              - patient_scores.json from the fixed subfolder (e.g. <fold_dir>/<es_key>/patient_scores.json)
            It then aggregates these results across folds.
            Returns:
              scores: dict with keys "default_night", "opt_night", and "opt_patient".
              (curves and cms could be handled similarly if needed.)
            """
        # Convert boolean early_stop_options to string keys.
        es_keys = ["early_stopping" if es else "no_early_stopping" for es in early_stop_options]

        # Initialize DataFrames/dicts.
        default_night_scores = pd.DataFrame()
        opt_night = {es: [] for es in es_keys}
        opt_patient = {es: [] for es in es_keys}

        curves = {lvl: {es: {'roc': {'fpr': [], 'tpr': [], 'auc': [], 'op_point': []},
                             'pr': {'rec': [], 'prec': [], 'f1_max': [], 'op_point_dist': [], 'op_point_f1': []}}
                        for es in es_keys} for lvl in ['night', 'patient']}

        cms = {lvl: {es: [] for es in es_keys} for lvl in ['night', 'patient']}

        for k, fold_dir in enumerate(fold_dirs):
            fold_name = f"fold_{k + 1}"
            # Read default model scores from the fold folder.
            try:
                default_scores = utils.read_from_json(fold_dir.joinpath("default_model_scores.json"))["valid"]
            except Exception as e:
                logger.warning(f"Could not read default_model_scores.json from {fold_dir}: {e}")
                continue
            # Convert the dictionary of metrics into a one-row DataFrame.
            df_default = pd.DataFrame({k: [v] for k, v in default_scores.items()}, index=[fold_name])
            default_night_scores = pd.concat([default_night_scores, df_default], axis=0)

            for es in es_keys:
                es_folder = fold_dir.joinpath(es)
                if not es_folder.exists():
                    logger.warning(f"Folder {es_folder} does not exist; skipping.")
                    continue

                for lvl in ['night', 'patient']:
                    score_file = es_folder.joinpath(
                        'night/night_scores.json' if lvl == 'night' else 'patient_scores.json')
                    if score_file.exists():
                        try:
                            scores = utils.read_from_json(score_file) if lvl == 'patient' \
                                else utils.read_from_json(score_file)['valid']
                            cm = scores.pop("cm", None)
                        except Exception as e:
                            logger.warning(f"Could not read {lvl}_scores.json from {score_file}: {e}")
                            continue

                        df_scores = pd.DataFrame({k: [v] for k, v in scores.items()}, index=[fold_name])
                        (opt_night if lvl == "night" else opt_patient)[es].append(df_scores)
                        if cm is not None:
                            cms[lvl][es].append(cm)

                # collect roc and pr interpolated curves
                for curve_type in ['roc', 'pr']:

                    for _lvl in ['night', 'patient']:

                        curve_dict = curves[_lvl][es][curve_type]

                        curve_file = es_folder.joinpath(
                            f'night/night_interp_{curve_type}_curve.json' if _lvl == 'night'
                            else f'patient_interp_{curve_type}_curve.json')

                        if curve_file.exists():
                            curve_data = utils.read_from_json(curve_file)

                            if curve_type == 'roc':
                                if not curve_dict['fpr']:
                                    curve_dict['fpr'] = curve_data['fpr']
                                curve_dict['tpr'].append(curve_data['tpr'])
                                curve_dict['auc'].append(curve_data.get('auc', None))  # Handle missing 'auc'
                                curve_dict['op_point'].append(curve_data.get('op_point', None))

                            elif curve_type == 'pr':
                                if not curve_dict['rec']:
                                    curve_dict['rec'] = curve_data['rec']
                                curve_dict['prec'].append(curve_data['prec'])
                                curve_dict['f1_max'].append(curve_data.get('f1_max', None))  # Handle missing 'f1_max'
                                curve_dict['op_point_dist'].append(curve_data.get('op_point', {}).get('dist', None))
                                curve_dict['op_point_f1'].append(curve_data.get('op_point', {}).get('f1', None))

        # For each early-stopping option, concatenate the per-fold DataFrames.
        for es in es_keys:
            if opt_night[es]:
                opt_night[es] = pd.concat(opt_night[es], axis=0)
            else:
                opt_night[es] = pd.DataFrame()
            if opt_patient[es]:
                opt_patient[es] = pd.concat(opt_patient[es], axis=0)
            else:
                opt_patient[es] = pd.DataFrame()

        scores = dict(default_night=default_night_scores, opt_night=opt_night, opt_patient=opt_patient)
        return scores, curves, cms

    @staticmethod
    def _draw_history_boxplots(scores: dict[str, pd.DataFrame], include_night: bool):
        figs_night, figs_patient = {}, {}

        def _recurse_dict_(d, key_path='', figs_dict=None):
            for key, value in d.items():
                new_key_path = f"{key_path}_{key}" if key_path else key
                if isinstance(value, dict):
                    _recurse_dict_(value, new_key_path, figs_dict)
                elif isinstance(value, pd.DataFrame):
                    value = value.T
                    figs_dict[new_key_path] = visualization.draw_cv_boxplot(
                        {metric: value.loc[metric, :].values for metric in value.index})
                    plt.close('all')

        if include_night:
            _recurse_dict_(scores['opt_night'], figs_dict=figs_night)
        _recurse_dict_(scores['opt_patient'], figs_dict=figs_patient)
        return figs_night, figs_patient

    @staticmethod
    def _draw_mean_curves(curves: dict[str, List], include_night: bool):
        figs_night, figs_patient = {}, {}

        def _recurse_dict_(d, key_path='', figs_dict=None):
            for key, value in d.items():
                new_key_path = f"{key_path}_{key}" if key_path else key
                first_value = list(value.values())[0]
                if isinstance(first_value, dict):
                    _recurse_dict_(value, new_key_path, figs_dict)
                elif isinstance(first_value, list):
                    # _mode = 'roc' if 'roc' in new_key_path else 'pr'
                    # figs[new_key_path] = plot_cv_summary.draw_cv_roc_or_pr_curve(value, _mode)
                    if 'roc' in new_key_path:
                        figs_dict[new_key_path] = visualization.draw_cv_roc_or_pr_curve(value, 'roc')
                    elif 'pr' in new_key_path:
                        figs_dict[new_key_path] = visualization.draw_cv_roc_or_pr_curve(value, 'pr')

        if include_night:
            _recurse_dict_(curves.get('night'), figs_dict=figs_night)
        _recurse_dict_(curves.get('patient'), figs_dict=figs_patient)

        return figs_night, figs_patient

    @contextmanager
    def _handle_parallel_logging(self):
        self.is_parallel = self.n_jobs != 1
        loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict
                   if 'aktiRBD' in name or '__main__' in name]

        previous_levels = {_lggr: logger.level for _lggr in loggers}

        if self.is_parallel:
            for _logger in loggers:
                _logger.setLevel(logging.WARNING)
            self.config.model.bayes_params.verbose = False
        else:
            self.config.model.bayes_params.verbose = True

        try:
            yield
        finally:
            if self.is_parallel:
                for _logger, level in previous_levels.items():
                    _logger.setLevel(level)

    @staticmethod
    @contextmanager
    def _manifest_guard(save_dir: Path, *, flavour: str, repeats: int, outer_folds_expected: int, inner_folds: int,
                        seeds: dict, dataset_summary: dict):
        """ Create/maintain *manifest.json* that tracks the progress of a Nested-CV run."""
        manifest_path = save_dir.joinpath('manifest.json')

        def __utc_now() -> str:
            return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        def __atomic_write(path: Path, payload: dict):
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, indent=2))
            tmp.replace(path)

        manifest = dict(run_id=str(uuid.uuid4()), flavour=flavour, started_at=__utc_now(), finished_at=None,
                        dataset_summary=dataset_summary, outer_folds_expected=outer_folds_expected,
                        inner_folds=inner_folds, repeats=repeats, seeds=seeds)
        __atomic_write(manifest_path, manifest)

        try:
            yield manifest  # give caller a mutable reference
        except Exception:
            raise  # manifest already indicates unfinished status
        else:
            manifest['finished_at'] = __utc_now()
            __atomic_write(manifest_path, manifest)

    @staticmethod
    def _count_unique_datasets(data: FeatureSet):
        return len(np.unique(data.dataset)) if data.dataset is not None else 1


class KFoldNestedCV(NestedCVBase):

    def _get_outer_splitter(self, seed: int):
        """Stratified group-aware k-kfold splitter."""
        return StratifiedGroupKFold(**self.config.nested_cv.outer_cv.dict(), random_state=seed)

    def _get_outer_groups(self, data: FeatureSet):
        """Group by patient ID."""
        return data.group


class LODONestedCV(NestedCVBase):

    def _get_outer_splitter(self, seed: int):
        """Leave one dataset out splitting."""
        return LeaveOneGroupOut()  # deterministic, no seed needed

    def _get_outer_groups(self, data: FeatureSet):
        """Group by datasets."""
        return data.dataset


if __name__ == '__main__':
    utils.setup_logging()

    _config = PipelineConfig.from_yaml(
        Path('/Users/david/Desktop/py_projects/aktiRBD_private/aktiRBD/src/aktiRBD/config/pipeline.yaml'))
    _save_path = Path(
        '/Users/david/Desktop/py_projects/aktiRBD_private/results/pipeline/run_2025-03-10_11h35m53s/nested_cv')

    nested_cv = NestedCV(_config, _save_path)
    nested_cv.eval()
