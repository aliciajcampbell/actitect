import logging
from typing import Union, Type
from pathlib import Path

from aktiRBD import utils
from aktiRBD.classifier.pipelines import BasePipeline
from aktiRBD.classifier.tools import ModelManager
from aktiRBD.classifier.tools.nested_cv_new import KFoldNestedCV, LODONestedCV
from aktiRBD.classifier.tools.feature_set import FeatureSet

logger = logging.getLogger(__name__)

__all__ = ['PooledTrainPipeline']


class PooledTrainPipeline(BasePipeline):

    def __str__(self):
        return f"PooledTrainPipeline(data={self.args.processed_data_dir})"

    def run(self):
        logger.info(f"starting {self} run.")

        # load the data
        train, _, _ = self._load_data()

        # run regular k-fold and a lodo cv
        for _cv in (KFoldNestedCV, LODONestedCV):
            self._run_nested_cv(_cv, train)

        # produce final model(s) for testing
        self._run_pretrain(train)

    def _run_nested_cv(self, cv_instance: Type[Union[KFoldNestedCV, LODONestedCV]], train: FeatureSet):
        _cv_name = cv_instance.__name__.replace("NestedCV", "")
        _save_path_cv = utils.check_make_dir(self.save_path.joinpath(_cv_name), True)

        config = self.config.copy()  # overwrite cv feature rankings path to have unique ones for kfold and lodo
        config.nested_cv.load_path_cv_feature_rankings = \
            Path(str(config.nested_cv.load_path_cv_feature_rankings).replace('cv_pooled', f'cv_pooled/{_cv_name}'))

        nested_cv = cv_instance(config, save_path=_save_path_cv)
        logger.info(f"[{_cv_name}]: starting nested cross-validation: {nested_cv}")
        with utils.Timer() as timer:
            nested_cv.fit(train.copy())
            logger.info(f"\n[{_cv_name}]: elapsed time with n_jobs={self.config.nested_cv.n_jobs}: {timer()}")
        nested_cv.eval()

    def _run_pretrain(self, train: FeatureSet, test: FeatureSet):
        _save_path_pretrain = utils.check_make_dir(self.save_path.joinpath('pretrain_hps'), True)
        model_manager_pretrain = ModelManager(self.config, _save_path_pretrain)
        model_manager_pretrain.pretrain(train.copy(), test.copy())
