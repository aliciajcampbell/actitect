from pathlib import Path
import logging
from typing import Union
from argparse import Namespace

from abc import ABC, abstractmethod
import numpy as np

from aktiRBD import utils
from aktiRBD.config import PipelineConfig, ExternalTestConfig
from aktiRBD.classifier.tools import DataLoader, NestedCV, ModelManager
from aktiRBD.classifier.tools.feature_set import FeatureSet

logger = logging.getLogger(__name__)

__all__ = ['TrainPipeline', 'TestPipeline']


class BasePipeline(ABC):

    def __init__(self, args: Namespace, config: Union[PipelineConfig, ExternalTestConfig], save_path: Path):
        self.args = args
        self.config = config
        self.save_path = save_path

    def _load_data(self):
        data_loader = DataLoader(self.args.processed_data_dir, self.args.meta_file, **self.config.data.loader.dict())
        return data_loader.get_train_test_data(agg_level=self.config.data.agg_level)

    @abstractmethod
    def run(self):
        raise NotImplementedError('abstractmethod')

    @abstractmethod
    def __str__(self):
        raise NotImplementedError('abstractmethod')


class TrainPipeline(BasePipeline):

    def __str__(self):
        return f"TrainPipeline(data={self.args.processed_data_dir})"

    def run(self):
        logger.info(f"starting {self} run.")

        # load the data
        train, test, _ = self._load_data()

        # fit nested cross-validation with hp optimization:
        self._run_nested_cv(train)

        # produce final model(s) for testing
        self._run_pretrain(train, test)  # 'test' only needed to create combined training set (for external test only)

        # load pretrained models and use for evaluation
        self._run_eval(test)

    def _run_nested_cv(self, train: FeatureSet):
        _save_path_cv = utils.check_make_dir(self.save_path.joinpath('nested_cv'), True)
        nested_cv = NestedCV(self.config, save_path=_save_path_cv)
        logger.info(f"starting nested cross-validation: {nested_cv}")
        with utils.Timer() as timer:
            nested_cv.fit(train.copy())
            logger.info(f"\n elapsed time with n_jobs={self.config.nested_cv.n_jobs}: {timer()}")
        nested_cv.eval()

    def _run_pretrain(self, train: FeatureSet, test: FeatureSet):
        _save_path_pretrain = utils.check_make_dir(self.save_path.joinpath('pretrain_hps'), True)
        model_manager_pretrain = ModelManager(self.config, _save_path_pretrain)
        model_manager_pretrain.pretrain(train.copy(), test.copy())

    def _run_eval(self, test: FeatureSet):
        _save_path_test_cgn = utils.check_make_dir(self.save_path.joinpath('test_cgn'), True)
        _load_path_models = self.save_path.joinpath('models/models_cgn_train.joblib')
        model_manager_eval = ModelManager(self.config, _save_path_test_cgn)
        model_manager_eval.eval(test, _load_path_models)


class TestPipeline(BasePipeline):

    def __str__(self):
        return f"TestPipeline(data={self.args.processed_data_dir})"

    def run(self):

        logger.info(f"starting {self} run.")

        # load the data
        _, test, __ = self._load_data()
        if _.x.shape[0] > 0:
            logger.warning(f" training set in external test mode has {_.x.shape[0]} assigned samples.")

        for _path_models in self.config.pretrained_model_dirs:
            # assert _
            if self.args.test:
                self._run_eval(test, _path_models)

            else:  # elif args.predict:
                # inference on external data without labels
                raise NotImplementedError('todo')

    def _run_eval(self, test: FeatureSet, model_dir: Path, case: str = None):

        # 'case' argument deprecated, was needed for Oxford testing with dominant vs non-dominant hand...
        _save_path = self.save_path.joinpath('external/eval') if case is None \
            else self.save_path.joinpath(f"external/eval/{case}")
        if case:
            logger.info(f'starting inference for {case} case:')

        _has_multiple_models = len(self.config.pretrained_model_dirs) > 1
        if _has_multiple_models:
            model_dir = Path(model_dir) if isinstance(model_dir, str) else model_dir
            _save_path = _save_path.joinpath(model_dir.stem)

        _save_path = utils.check_make_dir(_save_path, use_existing=True, verbose=False)
        eval_manager = ModelManager(self.config, _save_path)
        eval_manager.eval(test, model_dir)

    def _run_predict(self, test: FeatureSet, model_dir: Path):
        _save_path = utils.check_make_dir(self.save_path.joinpath('external/predict'), True, False)
        predict_manager = ModelManager(self.config, _save_path)
        predict_manager.predict(test, model_dir)


if __name__ == '__main__':

    import argparse
    import datetime
    import logging
    import sys
    from pathlib import Path

    from aktiRBD import __file__ as _aktirbd_root_file
    from aktiRBD import utils
    from aktiRBD.classifier.tools.pipeline import TrainPipeline, TestPipeline
    from aktiRBD.config import PipelineConfig, ExternalTestConfig

    logger = logging.getLogger(__name__)
    utils.setup_logging()


    def _run_setup():
        def _parse_args():
            parser = argparse.ArgumentParser(
                prog='aktiRBD',
                description='Entry point to perform RBD analysis from actigraphy data. Call --help or -h for details.',
                formatter_class=argparse.RawTextHelpFormatter)

            # arguments that define the operation mode of the script:
            group = parser.add_mutually_exclusive_group(required=False)  # to enable default
            group.add_argument('--train_eval', action='store_true', help='Run the entire training pipeline on '
                                                                         'cologne data: nested cv on train set, train model'
                                                                         ' on entire train set, eval on test set. ')
            group.add_argument('--test', action='store_true',
                               help='Use pretrained models to test on unseen data. (Default)')
            group.add_argument('--predict', action='store_true', help='Only run inference, no testing.')

            # Additional arguments for options within modes
            parser.add_argument('-r', '--root_dir', type=str, default=str(Path(__file__).resolve().parents[5]),
                                help="The root directory containing 'data/' and 'aktiRBD/'.")

            parser.add_argument('-c', '--config_file', type=str,
                                default='./aktiRBD/src/aktiRBD/config/pipeline.yaml',  # rel. to root dir
                                help='full path (rel. to root) of the config .yaml file defining preprocessing settings.')

            parser.add_argument('-d', '--processed_data_dir', type=str,
                                default='./data/processed/',  # rel. to root dir
                                help='directory (rel. to root) containing the pre-processed data.')

            parser.add_argument('-m', '--meta_file', type=str,
                                default='./data/raw/meta/metadata.csv',  # rel. to root dir
                                help="meta data file path (rel. to root) that contains list of patients. Has to contain "
                                     "the class labels except for execution in 'predict' mode. ")

            _args = parser.parse_args()

            # Set default operation mode:
            if not any([_args.train_eval, _args.test, _args.predict]):
                _args.train_eval = True

            # cast path-like args into PosixPath types
            _args.root_dir = Path(_args.root_dir)
            _args.config_file = _args.root_dir.joinpath(_args.config_file)
            _args.processed_data_dir = _args.root_dir.joinpath(_args.processed_data_dir)
            _args.meta_file = _args.root_dir.joinpath(_args.meta_file)

            assert (_args.root_dir.is_dir() and _args.root_dir.joinpath('data').is_dir() and
                    _args.root_dir.joinpath('aktiRBD').is_dir()), \
                (f" '--root_dir' ('-r') ({_args.root_dir}) is not a dir "
                 f"or does not contain sub-dirs './data' and './aktiRBD'")
            assert _args.config_file.is_file(), \
                f" '--config_file' (-'c') is not a file. ({_args.config_file})"

            return _args

        args = _parse_args()
        _experiment_id = '2025-03-10_16h12m47s'
        root_save_path = utils.check_make_dir(Path(args.root_dir).joinpath(f"results/pipeline/run_{_experiment_id}"),
                                              use_existing=True)
        utils.setup_logging(log_file_path=root_save_path.joinpath('log'))
        if args.train_eval:
            config = PipelineConfig.from_yaml(args.config_file)
        elif args.test or args.predict:
            config = ExternalTestConfig.from_yaml(args.config_file)
            config.pretrained_model_dirs = [args.root_dir.joinpath(_dir) for _dir in config.pretrained_model_dirs]
        else:
            raise ValueError(f"must choose one of args.train_eval, args.test or args.predict.")

        _case_ident = f"{config.data.loader.feature_dir.replace('/features', '')}" + "_" + "_".join(
            config.data.loader.aggregation) + f"/{sys.platform}" + f"/scale_{config.data.processing.scaling_order}"
        if config.data.processing.scaling_order == 'before_ranking':
            _case_ident += f"/{config.data.processing.scaler}_scaler"

        if args.train_eval:
            if not config.nested_cv.load_path_cv_feature_rankings:
                config.nested_cv.load_path_cv_feature_rankings = args.root_dir.joinpath(
                    f"results/feat_rank/cv/{_case_ident}")
            else:
                config.nested_cv.load_path_cv_feature_rankings = Path(config.nested_cv.load_path_cv_feature_rankings)

            if not config.final_model.load_path_feature_rankings:
                config.final_model.load_path_feature_rankings = args.root_dir.joinpath(
                    f"results/feat_rank/full/{_case_ident}")
            else:
                config.final_model.load_path_feature_rankings = Path(config.final_model.load_path_feature_rankings)

            if not config.final_model.nested_cv_path:
                config.final_model.nested_cv_path = root_save_path.joinpath('nested_cv')
            else:
                config.final_model.nested_cv_path = Path(config.final_model.nested_cv_path)

            if not config.final_model.save_path_models:
                config.final_model.save_path_models = root_save_path.joinpath('models')
                if config.final_model.overwrite_final_repo_models:
                    config.final_model.save_path_models = [config.final_model.save_path_models]
                    config.final_model.save_path_models.append(
                        Path(_aktirbd_root_file).parent.joinpath(f"models/pretrained/"))
            else:
                config.final_model.save_path_models = Path(config.final_model.save_path_models)

        # save the settings
        _dump_dict = {'command line options': vars(args), 'config': config.dict()}
        utils.dump_to_json(_dump_dict, root_save_path.joinpath('settings.json'))

        return args, config, root_save_path


    args, config, save_path = _run_setup()

    pipeline = TrainPipeline(args, config, save_path)
    pipeline.run()
