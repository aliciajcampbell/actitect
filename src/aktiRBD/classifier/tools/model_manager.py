import logging
import re
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Union

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from skopt.plots import plot_convergence, plot_evaluations, plot_objective
from skopt.space import Integer, Real, Categorical

from aktiRBD import utils
from aktiRBD.classifier.models import ModelFactory
from aktiRBD.classifier.tools import BayesianOptCV
from aktiRBD.classifier.tools.classification_threshold import get_night_and_patient_threshold
from aktiRBD.classifier.tools.evaluator import Evaluator
from aktiRBD.classifier.tools.feature_set import FeatureSet
from aktiRBD.classifier.tools.probability_calibration import CustomCalibratedClassifierCV
from aktiRBD.config import PipelineConfig, ExternalTestConfig, ExperimentConfig, ModelConfig

__all__ = ['ModelManager']
logger = logging.getLogger(__name__)


class ModelManager:
    @dataclass
    class HpSetup:  # model dependent HPs
        name: str
        hyperparameters: Dict[str, any]
        selected_features: List[str]

    def __init__(self, config: Union[PipelineConfig, ExternalTestConfig], save_path: Path):

        self.config = config
        self.save_path = save_path
        self.random_state = config.random_state

        if isinstance(config, PipelineConfig):  # training is triggered
            self.process_kwargs = self.config.data.processing.dict()
            self.process_kwargs.update({'smote_seed': self.random_state})
            self.n_jobs = config.final_model.n_jobs
            self.rank_kwargs = {
                'root_dir': self.config.final_model.load_path_feature_rankings,
                'data_config': self.config.data,
                'n_jobs': self.n_jobs}
            self._log_night_level = self.config.final_model.log_night_level
            self._output_patient_csv = self.config.final_model.output_patient_csv

            if any(exp.early_stopping for exp in self.config.final_model.experiments):
                assert self.config.model.which == 'xgboost', \
                    f"'early_stopping' option only available for 'xgboost' model."

            if any(exp.hp_setup_name != 'bayes_opt' for exp in self.config.final_model.experiments):
                assert self.config.final_model.nested_cv_path.is_dir(), \
                    f"specified nested_cv run at {self.config.final_model.nested_cv_path.resolve()} not found."
                logger.info(f"using nested_cv run at {self.config.final_model.nested_cv_path} for hp* estimation.")

        else:  # only used for inference
            self.process_kwargs = self.n_jobs = self.rank_kwargs = self.early_stopping_options = None
            self._log_night_level = self.config.log_night_eval
            self._output_patient_csv = self.config.output_patient_csv

        self.model_setup = None
        self.current_context = None

    def __str__(self):
        return f"ModelManager(random_state={self.random_state})"

    def eval(self, test: FeatureSet, model_dict_file: Path, train: FeatureSet = None, filter_min_nights: bool = True):

        assert model_dict_file.is_file() and model_dict_file.suffix == '.joblib', \
            f"Saved model file '{model_dict_file}' does not exist or is not a '.joblib' file."
        model_dict = joblib.load(model_dict_file)

        with (utils.custom_tqdm(total=len(model_dict)) as pbar):
            pbar.set_description("[PROGRESS]: Inference")
            for _iteration in self._inference_iterator(test, model_dict, train):
                _save_path, _exp, model, _, _, thresholds, train_processed, test_processed = _iteration

                if filter_min_nights:
                    _min_n_nights = self.config.min_patient_nights_eval if isinstance(self.config, ExternalTestConfig) \
                        else self.config.final_model.min_patient_nights_eval
                    for _valid_set, _save_tag in [
                        (test_processed, 'all'),
                        (test_processed.filter_patients_min_nights(_min_n_nights), f'min_{_min_n_nights}_nights'),
                    ]:
                        _out_dir = utils.check_make_dir(_save_path.joinpath(_save_tag), True)
                        ev = Evaluator(
                            _out_dir, _exp, thresholds, output_patient_csv=self._output_patient_csv, cv_mode=False)
                        ev.evaluate(train_data=None, valid_data=_valid_set, generate_night_output=self._log_night_level)
                else:
                    _out_dir = utils.check_make_dir(_save_path, True)
                    ev = Evaluator(
                        _out_dir, _exp, thresholds, output_patient_csv=self._output_patient_csv, cv_mode=False)
                    ev.evaluate(train_data=None, valid_data=test_processed, generate_night_output=self._log_night_level)

                pbar.update(1)

    def pretrain(self, train: FeatureSet, test: FeatureSet = None, *, dataset_save_tag: str):
        """Like .eval() but will save the models to a given directory instead of evaluating them directly.
         Mainly used to create the final models for external testing."""
        # init model
        self.model_setup = self._get_model_setup(self.config.model, train.y, self.random_state)
        chosen_hp_setups = {exp.hp_setup_name for exp in self.config.final_model.experiments}

        # if specified, create a merged cologne training set (train + test), can only be used for external data testing
        train_full_processed, hp_setups_merged = None, []
        if self.config.final_model.include_pretrain_merged and test is not None:  # <- guard added
            train_full = train.merge(test)
            train_full_processed = train_full.copy().fit_transform(rank_kwargs=self.rank_kwargs,
                                                                   **self.process_kwargs)
            with self._set_current_context(f'pretrain_{dataset_save_tag}_merged'):
                hp_setups_merged = self._get_hp_setups(train_full_processed, chosen_hp_setups)

        # precess features: scaling, ranking, smote (always call after combined set is created!)
        train_processed = train.copy().fit_transform(rank_kwargs=self.rank_kwargs, **self.process_kwargs)

        with self._set_current_context(f'pretrain_{dataset_save_tag}'):
            hp_setups = self._get_hp_setups(train_processed, chosen_hp_setups)

        # train the model(s) and dump to .joblib file
        model_dict_cgn, model_dict_full = {}, {}
        total_iterations = len(
            self.config.final_model.experiments) * (
                               2 if self.config.final_model.include_pretrain_merged and test is not None else 1)

        with utils.custom_tqdm(total=total_iterations) as pbar:
            pbar.set_description(f"[PROGRESS]: Pretraining")

            # first loop for cologne training set
            for exp in self.config.final_model.experiments:
                _hp_setup = hp_setups.get(exp.hp_setup_name)
                if _hp_setup is None:
                    logger.warning(f"Skipping experiment '{exp.name}':"
                                   f"hp strategy '{exp.hp_setup_name}' not computed.")
                    continue
                model_dict_cgn.update(self._train_model(train_processed, _hp_setup, experiment=exp, pbar=pbar))
                pbar.update(1)

            # second loop for full pretrain if enabled
            if self.config.final_model.include_pretrain_merged and test is not None:
                for exp in self.config.final_model.experiments:
                    _hp_setup_full = hp_setups_merged.get(exp.hp_setup_name)
                    if _hp_setup_full is None:
                        logger.warning(f"Skipping experiment '{exp.name}' in full pretrain:"
                                       f"HP setup '{exp.hp_setup_name}' not computed.")
                        continue
                    model_dict_full.update(
                        self._train_model(train_full_processed, _hp_setup_full, experiment=exp, pbar=pbar))
                    pbar.update(1)

        model_save_path_list = self.config.final_model.save_path_models
        if not isinstance(model_save_path_list, list):
            model_save_path_list = list(model_save_path_list)

        for _model_save_path in model_save_path_list:
            _model_save_path = utils.check_make_dir(_model_save_path, use_existing=True, verbose=False)
            joblib.dump(model_dict_cgn, _model_save_path.joinpath(f'models_{dataset_save_tag}.joblib'))
            if self.config.final_model.include_pretrain_merged and test is not None and model_dict_full:
                joblib.dump(model_dict_full, _model_save_path.joinpath(f'models_{dataset_save_tag}_merged.joblib'))
            logger.info(f"successfully dumped models to {_model_save_path}")

    def predict(self, test: FeatureSet, model_dict_file: Path):
        """with utils.custom_tqdm(total=sum(1 for _ in utils.nested_dict_iterator(
                joblib.load(model_dict_file), iteration_depth=4))) as pbar:
            pbar.set_description("[PROGRESS]: Inference")
            for _iteration in self._inference_iterator(test, model_dict_file, train):
                _save_path, _es_suffix, model, _, _, thresholds, train_processed, _ = _iteration

                # save the predictions and thresholds...
                per_night_df = pd.DataFrame({
                    'true': test_processed.y, 'prob': test_processed.prob, 'group': test_processed.group})

                for night_thres in thresholds['night_thresholds']:
                    _y_pred_night = classify_with_threshold(test_processed.prob, night_thres.value)
                    per_night_df[f"pred_{night_thres.name} ({night_thres.value})"] = _y_pred_night

                    patient_threshold = patient_threshold.get(night_thres.name)  # use same approach
                    _agg_kwargs = {'mean_prob_threshold': patient_threshold.value, 'majority_vote_frac': 2}
                    per_patient_df = aggregate_night_predictions_to_patient_level(
                        tet_processed.prob, y_pred=_y_pred_night, kwargs=_agg_kwargs)

                    print(per_night_df)
                    print(per_patient_df)
                    # TODO!
                    raise SystemExit("todo: add per patient predictions to dataframe and save it...")"""
        raise NotImplementedError('todo')

    def _train_model(self, train: FeatureSet, hp_setup: HpSetup, experiment: ExperimentConfig,
                     test: FeatureSet = None, pbar=None):
        """ Train the model for each HpSetup instance. Different modes to create model either for Cologne or
        external testing scenario.
        Parameters:
            :param hp_setup: (HpSetup) the choice of hyperparameters.
            :param train: (FeatureSet) Cologne training.
            :param experiment: (Experiment) defining model-independent training HPs.
            :param test: (FeatureSet, Optional) Cologne testing data. Only needed in eval mode.
            :param pbar: (tqdm.pbar, Optional) If provided, will log some infos within pbar.
        Returns:
            :return: (Union[None, Dict]) in pretrain modes will return a dict containing the models to save them, else
                None, and it will save the evaluation results directly. """
        train = train.select_features(hp_setup.selected_features)
        if test is not None:
            test = test.select_features(hp_setup.selected_features)
        _eval_set = None

        _model_dict = {}
        _eval_set = [(train.x, train.y), (test.x, test.y)] if test is not None else [(train.x, train.y)]

        # set up the model with specified HP*
        hps = self.model_setup.bayesian_fixed_params.copy()
        hps.update(hp_setup.hyperparameters)
        _ = hps.pop('top_k_feats')

        if pbar:
            _postfix_dict = {'name': hp_setup.name}
            pbar.set_postfix(_postfix_dict)
        else:
            logger.info(f"{hp_setup.name}: setting model params to "
                        f"{utils.fmt_dict(hp_setup.hyperparameters, 3)}.")

        # initialize model with computed hps
        opt_model = self.model_setup.model().set_params(**hps)

        # apply early stopping settings based on the experiment.
        if experiment.early_stopping:
            opt_model.set_params(**{'early_stopping_rounds': 10, 'eval_metric': ['logloss']})
        else:
            opt_model.set_params(**{'early_stopping_rounds': None, 'eval_metric': ['logloss']})

        # apply calibration if specified in the experiment.
        if experiment.calibration != 'none':
            _sgkf = StratifiedGroupKFold(
                **self.config.final_model.calibration_cv.dict(), random_state=self.random_state)
            calibrated_cv = CustomCalibratedClassifierCV(
                estimator=opt_model, method=experiment.calibration, cv=_sgkf,
                pass_fold_as_eval_set=True, n_jobs=self.n_jobs)
            logger.info(f"Training model with {experiment.calibration} calibration for experiment '{experiment.name}'.")
            opt_model = calibrated_cv.fit(train.x, train.y, groups=train.group, verbose=False)
        else:
            opt_model.fit(train.x, train.y, eval_set=_eval_set, verbose=False)

        # use training data to determine classification thresholds.
        train.prob = opt_model.predict_proba(train.x)[:, 1]
        night_threshold, patient_threshold = get_night_and_patient_threshold(train, experiment.threshold)

        result = {
            'model': opt_model,  # the model
            'hp_setup': hp_setup,  # the model dependent HPs
            'processing': train.process_params,  # params uses for processing (e.g. normalization)
            'thresholds': {  # the classification thresholds
                'night': night_threshold,
                'patient': patient_threshold
            },
            'experiment': experiment  # mode independent HPs
        }

        return {experiment.name: result}

    def _inference_iterator(self, test: FeatureSet, model_dict: dict, train: FeatureSet = None):
        """Shared logic for .predict() and .eval(): loads models, processes the test (and train) data,
        runs prediction, and yields the common variables."""
        assert model_dict, f"model_dict has to be provided but got '{model_dict}'."
        for exp_name, model_data in model_dict.items():
            _save_path = utils.check_make_dir(self.save_path.joinpath(exp_name), True, True)
            model = model_data['model']
            hp_setup = model_data['hp_setup']
            process_params = model_data['processing']
            thresholds = model_data['thresholds']
            experiment = model_data['experiment']

            # Preprocess test data and (if provided) train data using saved process_params
            test_processed = test.copy().transform(process_params)
            test_processed = test_processed.select_features(hp_setup.selected_features)
            if train is not None:
                train_processed = train.copy().transform(process_params)
                train_processed = train_processed.select_features(hp_setup.selected_features)
            else:
                train_processed = None

            # run inference, add probability predictions
            test_processed.prob = model.predict_proba(test_processed.x)[:, 1]
            if train_processed is not None:
                train_processed.prob = model.predict_proba(train_processed.x)[:, 1]

            yield _save_path, experiment, model, hp_setup, process_params, thresholds, train_processed, test_processed

    def _get_hp_setups(self, train: FeatureSet, chosen_setups: List[str]):

        hp_funcs = {
            'bayes_opt': lambda: self._get_bayes_opt_hps(train),
            # HP* based on averages over nested-cv results
            'nested_cv_avg_full_rank': lambda: self._get_nested_cv_avg_hps(
                train.feat_rank, ranking_mode='full_data', weighted=False),
            'nested_cv_avg_fold_avg': lambda: self._get_nested_cv_avg_hps(
                train.feat_rank, ranking_mode='fold_avg', weighted=False),
            'nested_cv_avg_weighted_full_rank': lambda: self._get_nested_cv_avg_hps(
                train.feat_rank, ranking_mode='full_data', weighted=True),
            'nested_cv_avg_weighted_fold_avg': lambda: self._get_nested_cv_avg_hps(
                train.feat_rank, ranking_mode='fold_avg', weighted=True),
            # HP* based on best performing folds of nested-cv results
            'nested_cv_best_full_rank': lambda: self._get_nested_cv_best_hps(
                train.feat_rank, ranking_mode='full_data', use_top_fold_only=True),
            'nested_cv_best_fold_avg': lambda: self._get_nested_cv_best_hps(
                train.feat_rank, ranking_mode='fold_avg', use_top_fold_only=True),
            'nested_cv_top_10_perc_avg_full_rank': lambda: self._get_nested_cv_best_hps(
                train.feat_rank, ranking_mode='full_data', use_top_fold_only=False, top_frac=.1),
            'nested_cv_top_10_perc_avg_fold_avg': lambda: self._get_nested_cv_best_hps(
                train.feat_rank, ranking_mode='fold_avg', use_top_fold_only=False, top_frac=.1),
        }

        if chosen_setups:
            hp_funcs = {name: func for name, func in hp_funcs.items() if name in chosen_setups}

        return {name: func() for name, func in hp_funcs.items()}

    @staticmethod
    def _get_model_setup(model_cfg: ModelConfig, y_train: np.ndarray, random_state: int):
        _n_hc_train_outer, _n_rbd_train_outer = np.unique(y_train, return_counts=True)[1]
        _cls_balance_outer = _n_hc_train_outer / _n_rbd_train_outer
        model_factory = ModelFactory(model_cfg.which, cls_balance=_cls_balance_outer, seed=random_state,
            top_k_cfg=model_cfg.feature_selection.top_k_feats)
        return model_factory.build()

    def _get_bayes_opt_hps(self, train: FeatureSet, plot_search_history: bool = True):
        """ Perform Bayesian optimization on full training set to determine the best hyperparameters and selected
        features. Uses ranking of features computed over full training set.
        :param train: (FeatureSet) The training data.
        :param plot_search_history: bool, optional
            Whether to plot and save the Bayesian optimization search history, by default True.
        :returns: HpSetup
            An instance of `HpSetup` containing the strategy name ('bayes_opt'), the optimized hyperparameters,
            and the selected feature names.
        """

        bayesian_opt = BayesianOptCV(self.model_setup, train.feat_rank, self.random_state)

        fit_params = self.config.final_model.bayes_params.dict().copy()
        fit_params.update({'cv_params': {'group': train.group, 'n_folds': self.config.final_model.bayes_cv.n_splits,
                                         'shuffle': self.config.final_model.bayes_cv.shuffle, 'verbose': False},
                           'n_jobs': self.n_jobs})

        y_strat = train.get_strat_labels(self.config.final_model.stratify_by_dataset_if_pooled)
        results = bayesian_opt.fit(train.x, train.y, y_strat, **fit_params)
        opt_params = self.format_hp_dict({dim.name: val
                                          for dim, val in zip(self.model_setup.bayesian_param_space, results.x)},
                                         self.model_setup.bayesian_param_space)
        _selected_feats, _info_dict_feats = self._get_selected_feat_list(
            train.feat_rank, opt_params['top_k_feats'], 'full_data')

        logger.info(f"best score: {-results.fun:.3f} with params {utils.fmt_dict(opt_params, 3)}")

        # saving and plotting:
        _name = f"{self.current_context}/bayes_opt"
        save_path_bay_opt = utils.check_make_dir(self.save_path.joinpath(_name), use_existing=True, verbose=False)
        hp_summary = {'strategy': {'bayes_opt': True, 'rank': 'full_data'},
                      'best_params': opt_params,
                      'best score': -results.fun}
        hp_summary.update(_info_dict_feats)
        for filename, _dict in zip(
                ['hp_summary', 'fixed_params', 'param_space'],
                [hp_summary, self.model_setup.bayesian_fixed_params, self.model_setup.bayesian_param_space]
        ):
            utils.dump_to_json(_dict, save_path_bay_opt.joinpath(f"{filename}.json"))

        if plot_search_history:  # plot hp search:
            plt.close('all')
            _plotting_funcs = {
                'convergence': (plot_convergence, {}),
                'evaluations': (plot_evaluations, {'cmap': 'YlGnBu_r'}),
                'objective': (plot_objective, {'sample_source': 'result', 'cmap': 'YlGnBu'})
            }
            for _fig_name, (func, kwargs) in _plotting_funcs.items():
                for _ext in ('png', 'pdf'):
                    func(results, **kwargs).get_figure() \
                        .savefig(save_path_bay_opt.joinpath(f'{_fig_name}.{_ext}'), bbox_inches='tight')
                plt.close('all')

        return self.HpSetup(name=_name, hyperparameters=opt_params, selected_features=_selected_feats)

    def _get_nested_cv_best_hps(self, feature_ranking_map: dict, ranking_mode: str, use_top_fold_only: bool,
                                top_frac: float = .1):
        """Choose either the hyperparameters of the  one best fold from cross-validation
        or average over the top k percent folds.
        Parameters:
            :param use_top_fold_only: bool
               If True, use only the top-performing fold's hyperparameters and selected features.
               If False, use the top percentage of folds specified by top_percent and average their hyperparameters.
            :param ranking_mode: str
               Specifies the mode for selecting features. Options include 'fold_avg' for fold average or 'full_data'
               for full training data ranking.
            :param top_frac: float, optional
               The top percentage of folds to consider when creating the strategy. For example,
                top_frac=0.1 means using the top 10% of folds. Default is 0.1.
        :returns: HpSetup
           An HpSetup instance containing hyperparameters and selected features based on the best fold(s).
        :raises ValueError: If an invalid value is provided for top_percent. """

        _all_fold_dirs = sorted(list(self.config.final_model.nested_cv_path.joinpath('repeats').glob('**/*_fold_*')),
                                key=lambda x: int(re.search(r'repeat(\d+)_', str(x)).group(1)))

        folds_data = []
        for _fold_dir in _all_fold_dirs:
            hps = utils.read_from_json(_fold_dir.joinpath('bay_opt/best_params.json'))['best_params']
            scores = utils.read_from_json(_fold_dir.joinpath('early_stopping/night/night_scores.json'))
            mean_score = scores['valid']['mean']
            folds_data.append({'dir': _fold_dir, 'mean_score': mean_score, 'hps': hps})

        folds_df = pd.DataFrame(folds_data).sort_values(by='mean_score', ascending=False)

        if use_top_fold_only:  # best performing fold only
            _top_method = 'single_best'
            best_hps = folds_df.iloc[0]['hps']
            selected_folds = [folds_df.iloc[0]['dir']]
        else:  # use top_percent of folds and average
            _top_method = f'top_{top_frac * 100:.0f}_percent'
            if top_frac <= 0 or top_frac > 1:
                raise ValueError("Invalid value for top_percent. It should be a float between 0 and 1.")
            num_top_folds = max(1, int(len(folds_df) * top_frac))
            top_hps = folds_df.head(num_top_folds)['hps'].tolist()
            selected_folds = folds_df.head(num_top_folds)['dir'].tolist()
            best_hps = pd.DataFrame(top_hps).mean().to_dict()

        best_hps = self.format_hp_dict(best_hps.copy(), self.model_setup.bayesian_param_space)
        _name = f'{self.current_context}/best_hps/{_top_method}/{ranking_mode}_ranking'
        _selected_feats, _info_dict_feats = \
            self._get_selected_feat_list(feature_ranking_map, best_hps['top_k_feats'], ranking_mode)

        _dump_dict = {'strategy': {'top': _top_method,
                                   'rank': ranking_mode,
                                   'folds': [{'dir': str(fold_dir), 'mean_score': fold_score}
                                             for fold_dir, fold_score
                                             in zip(selected_folds, folds_df['mean_score'].tolist())]},
                      'hyperparameters': best_hps}
        _dump_dict.update(_info_dict_feats)
        _save_path_ = utils.check_make_dir(self.save_path.joinpath(_name), False, verbose=False)
        utils.dump_to_json(_dump_dict, _save_path_.joinpath('hp_summary.json'))

        return self.HpSetup(name=_name, hyperparameters=best_hps, selected_features=_selected_feats)

    def _get_nested_cv_avg_hps(self, feat_ranking_map: dict, ranking_mode: str, weighted: bool):
        """ Generate a hyperparameter strategy based on nested cross-validation average and selected features.
        Depending on the `weighted` parameter, either a simple mean or a weighted mean across all repeats/folds is used.
        The weight is given by the validation AUROC of the specific fold.
        Top-k selected features are based on either their frequency across all folds or their ranking over full data.

        :param weighted: bool
            Determines whether to use a weighted mean for hyperparameters. If True, the weighted mean is calculated;
            otherwise, a simple unweighted mean is used.

        :param ranking_mode: str
            Specifies the mode for selecting features. Should be 'fold_avg' for fold average or 'full_data' for full
            training data ranking.

        :returns: HpSetup
            An HpSetup instance containing the calculated hyperparameters and selected features based on the
            specified averaging and ranking mode.

        :raises AssertionError: If the top_k_feats value is greater than the number of available features.
        """

        if weighted:
            _avg_method = 'weighted_mean'
            _all_fold_dirs = sorted(list(
                self.config.final_model.nested_cv_path.joinpath('repeats').glob('**/*_fold_*')),
                key=lambda x: int(re.search(r'repeat(\d+)_', str(x)).group(1)))

            hps, weights = [], []
            for _fold_dir in _all_fold_dirs:
                hps.append(utils.read_from_json(_fold_dir.joinpath('bay_opt/best_params.json'))['best_params'])
                _scrs = utils.read_from_json(_fold_dir.joinpath('early_stopping/night/night_scores.json'))
                weights.append(_scrs['night_default']['valid']['mean'])  # 'night_default', 'night_roc_dist',

            mean_hps = self.format_hp_dict((pd.DataFrame(hps).T * pd.Series(weights) / sum(weights)).T.sum().to_dict(),
                                           self.model_setup.bayesian_param_space)

        else:
            _avg_method = 'mean'
            hp_summary = pd.read_csv(self.config.final_model.nested_cv_path.joinpath('hp_summary.csv'), index_col=0)
            mean_hps = self.format_hp_dict(hp_summary['mean'].to_dict(), self.model_setup.bayesian_param_space)

        _name = f'{self.current_context}/mean_hps/{_avg_method}/{ranking_mode}_ranking'

        _selected_feats, _info_dict_feats = self._get_selected_feat_list(
            feat_ranking_map, mean_hps['top_k_feats'], ranking_mode)
        _dump_dict = {'strategy': {'avg': _avg_method, 'rank': ranking_mode}, 'hyperparameters': mean_hps}
        _dump_dict.update(_info_dict_feats)
        _save_path_ = utils.check_make_dir(self.save_path.joinpath(_name), False, verbose=False)
        utils.dump_to_json(_dump_dict, _save_path_.joinpath('hp_summary.json'))

        return self.HpSetup(name=_name, hyperparameters=mean_hps, selected_features=_selected_feats)

    def _get_selected_feat_list(self, feat_rank_map: dict, top_k: int, ranking_mode: str):
        """ Select the top-k features based on the specified ranking mode and return both the list of selected features
        and an accompanying information dictionary.

        :param top_k: int
            The number of top features to select.
        :param ranking_mode: str
            Specifies the mode for selecting features. Options are:
            - 'fold_avg': Selects top-k features based on their frequency or importance across cross-validation folds.
            - 'full_data': Selects top-k features based on their ranking from the full training dataset.

        :returns: tuple
            A tuple containing:
            - list: The names of the selected features.
            - dict: A dictionary containing additional information about the selected features.
                If ranking_mode is 'fold_avg', this dictionary contains the feature names with their corresponding
                selection percentage across folds.
                If ranking_mode is 'full_data', this dictionary contains the feature names with their ranking.

        :raises AssertionError: If the top_k value exceeds the number of available features in the 'fold_avg' mode.
        :raises ValueError: If an invalid ranking_mode is provided.
        """

        if ranking_mode == 'fold_avg':  # selected top_k_features based on the fold history:
            top_k_summary = pd.read_csv(
                self.config.final_model.nested_cv_path.joinpath('top_k_summary.csv'), index_col=0)
            assert top_k < top_k_summary.shape[0], \
                (f"top_k_feats={top_k} is larger than number of features "
                 f"in 'top_k_summary' (={top_k_summary.shape[0]})")
            _selected_feat_top_k_summary_dict = top_k_summary.head(top_k)['sum'].to_dict()

            _info_dict = {f'selected_feats (n={top_k_summary.shape[1] - 1} fold percentage)': {
                name: np.round(value / (top_k_summary.shape[1] - 1) * 100, 1)
                for name, value in _selected_feat_top_k_summary_dict.items()}}
            return list(_selected_feat_top_k_summary_dict.keys()), _info_dict

        elif ranking_mode == 'full_data':  # selected top_k_features based on full training data ranking:
            _selected_feats_full_rank_dict = {
                _name: _val['rank'] for _name, _val in
                sorted(feat_rank_map.items(), key=lambda item: item[1]['rank'])
                if _val['rank'] <= top_k
            }
            _info_dict = {'selected_feats (full data rank)': _selected_feats_full_rank_dict}

            return list(_selected_feats_full_rank_dict.keys()), _info_dict
        else:
            raise ValueError(f"unknown 'ranking_mode' value encountered: {ranking_mode}."
                             f"Must be either 'fold_avg' or 'full_data'.")

    @staticmethod
    def format_hp_dict(hp_dict: dict, bayesian_param_space: list, float_decimals: int = 4):
        """
        Formats a dictionary of hyperparameters to ensure correct data types based on the parameter space.
        For example for averaged HPs.

        :param hp_dict: dict
            Dictionary containing hyperparameters with possibly incorrect data types.
        :param bayesian_param_space: list
            List of skopt.space parameters (Integer, Real, Categorical) defining the expected types for each hp.
        :param float_decimals: int, optional.
            Decimal precision of float values, default is 2, i.e. 0.01.

        :returns: dict
            A dictionary with hyperparameters correctly formatted to the expected data types.
        """
        formatted_hp_dict = hp_dict.copy()
        for param in bayesian_param_space:
            param_name = param.name
            if param_name in formatted_hp_dict:
                if isinstance(param, Integer):
                    formatted_hp_dict[param_name] = int(round(formatted_hp_dict[param_name]))
                elif isinstance(param, Real):
                    formatted_hp_dict[param_name] = np.round(formatted_hp_dict[param_name], float_decimals)
                elif isinstance(param, Categorical):
                    # Ensure the value is one of the allowed categories
                    if formatted_hp_dict[param_name] not in param.categories:
                        raise ValueError(
                            f"Value {formatted_hp_dict[param_name]} is not a valid category for {param_name}")
                    # No type conversion needed for categorical values, just a validation check

        return formatted_hp_dict

    @contextmanager
    def _set_current_context(self, context: str):
        """Context manager to temporarily set the current pretrain context."""
        previous_context = getattr(self, 'current_context', None)
        self.current_context = context
        try:
            yield
        finally:
            self.current_context = previous_context
