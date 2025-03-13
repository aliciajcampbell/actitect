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
