import logging
from pathlib import Path

from aktiRBD import utils
from aktiRBD.classifier.pipelines import BasePipeline
from aktiRBD.classifier.tools import ModelManager
from aktiRBD.classifier.tools.feature_set import FeatureSet

logger = logging.getLogger(__name__)

__all__ = ['TestPipeline']


class TestPipeline(BasePipeline):

    def __str__(self):
        return f"TestPipeline(data={self.args.processed_data_dir})"

    def run(self):

        logger.info(f"starting {self} run.")

        # load the data
        _, test, __ = self._load_data()
        if _.x.shape[0] > 0:
            logger.warning(f"training set in external test mode has {_.x.shape[0]} assigned samples.")

        for _path_models in self.config.pretrained_model_dirs:
            if self.args.test:
                self._run_eval(test, _path_models)

            else:  # elif args.predict:
                # inference on external data without labels
                raise NotImplementedError('todo')

    def _run_eval(self, test: FeatureSet, model_dir: Path, case: str = None):

        # 'case' argument deprecated, was needed for Oxford testing with dominant vs non-dominant hand...
        _save_path = self.save_path.joinpath(f"{self.args.dataset_tag}/eval") if case is None \
            else self.save_path.joinpath(f"{self.args.dataset_tag}/eval/{case}")
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
