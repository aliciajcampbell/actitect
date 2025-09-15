import logging

from aktiRBD import utils
from aktiRBD.classifier.pipelines import BasePipeline
from aktiRBD.classifier.tools import ModelManager
# from aktiRBD.classifier.tools.nested_cv_legacy import NestedCV  # todo: use refactored Abstract class!
from aktiRBD.classifier.tools import KFoldNestedCV
from aktiRBD.classifier.tools.feature_set import FeatureSet

logger = logging.getLogger(__name__)

__all__ = ['TrainPipeline']


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
        if not self.config.final_model.skip_pretrain:
            self._run_pretrain(train, test)  # 'test' only needed to create combined cgn training set (ext. test only)
            # load pretrained models and use for evaluation
            self._run_eval(test)
        else:
            logger.warning("skipping pretraining of final model, set 'config.final_model.skip_pretrain=True'.")

    def _run_nested_cv(self, train: FeatureSet):
        _save_path_cv = utils.check_make_dir(self.save_path.joinpath('nested_cv'), True)
        nested_cv = KFoldNestedCV(self.config, save_path=_save_path_cv)
        logger.info(f"starting nested cross-validation: {nested_cv}")
        with utils.Timer() as timer:
            nested_cv.fit(train.copy())
            logger.info(f"\n elapsed time with n_jobs={self.config.nested_cv.n_jobs}: {timer()}")
        nested_cv.eval()

    def _run_pretrain(self, train: FeatureSet, test: FeatureSet):
        _save_path_pretrain = utils.check_make_dir(self.save_path.joinpath('pretrain_hps'), True)
        model_manager_pretrain = ModelManager(self.config, _save_path_pretrain)
        model_manager_pretrain.pretrain(train.copy(), test.copy(), dataset_save_tag='cgn')

    def _run_eval(self, test: FeatureSet):
        _save_path_test_cgn = utils.check_make_dir(self.save_path.joinpath('test_cgn'), True)
        _load_path_models = self.save_path.joinpath('models/models_cgn.joblib')
        model_manager_eval = ModelManager(self.config, _save_path_test_cgn)
        model_manager_eval.eval(test, _load_path_models)
