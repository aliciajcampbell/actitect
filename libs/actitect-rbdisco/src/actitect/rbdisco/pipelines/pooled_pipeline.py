import logging
from argparse import Namespace
from pathlib import Path
from typing import Type, Union

import numpy as np

from actitect import utils
from actitect.classifier.pipelines import BasePipeline
from actitect.classifier.tools import ModelManager
from actitect.classifier.tools.feature_set import FeatureSet
from actitect.classifier.tools.nested_cv import KFoldNestedCV, LODONestedCV
from actitect.config import PipelineConfig, ExternalTestConfig

logger = logging.getLogger(__name__)

__all__ = ['PooledTrainPipeline']


class PooledTrainPipeline(BasePipeline):

    def __init__(self, args: Namespace, config: Union[PipelineConfig, ExternalTestConfig], save_path: Path,
                 skip_kfold: bool = True):
        super().__init__(args, config, save_path)
        self.ds_suffix = None
        self.temp_config = config.copy()

        # caches to avoid stacking modifications
        self._cv_rankings_base = Path(self.temp_config.nested_cv.load_path_cv_feature_rankings)
        self._nested_cv_base = Path(self.temp_config.final_model.nested_cv_path)
        self._full_rankings_base = Path(self.temp_config.final_model.load_path_feature_rankings)
        self.skip_kfold = skip_kfold

    def __str__(self):
        return f"PooledTrainPipeline(data={self.args.processed_data_dir})"

    def run(self):
        logger.info(f"starting {self} run.")

        # load the data
        train, _, _ = self._load_data()
        self.ds_suffix = "_".join(map(str, sorted(np.unique(train.dataset))))

        # run regular k-fold and a lodo cv
        cv_flavours = [LODONestedCV, KFoldNestedCV] if not self.skip_kfold else [LODONestedCV]
        logger.info(f"running pooled cv with flavour(s): {cv_flavours}")
        for _cv in cv_flavours:
            self._run_nested_cv(_cv, train)

        # produce final model(s) for testing
        if not self.temp_config.final_model.skip_pretrain:
            self._run_pretrain(train)
        else:
            logger.warning("skipping pretraining of final model, set 'config.final_model.skip_pretrain=True'.")

        # update and dump config changes
        self._reconcile_and_update_config(
            settings_path=self.save_path.joinpath('settings.json'),
            updated_config=self.temp_config.dict()
        )

    def _run_nested_cv(self, cv_instance: Type[Union[KFoldNestedCV, LODONestedCV]], train: FeatureSet):
        _cv_name = cv_instance.__name__.replace("NestedCV", "")
        _save_path_cv = utils.check_make_dir(self.save_path.joinpath(f"{_cv_name}_{self.ds_suffix}"), True)

        # always rewrite from cached base
        self.temp_config.nested_cv.load_path_cv_feature_rankings = Path(
            str(self._cv_rankings_base).replace(
                'cv_pooled', f'cv_pooled/{_cv_name}_{self.ds_suffix}'
            )
        )

        nested_cv = cv_instance(
            self.temp_config,
            save_path=_save_path_cv,
            calibration=self.config.nested_cv.default_experiment.calibration
        )
        logger.info(f"[{_cv_name}]: starting nested cross-validation: {nested_cv}")
        with utils.Timer() as timer:
            nested_cv.fit(train.copy())
            logger.info(f"\n[{_cv_name}]: elapsed time with n_jobs={self.config.nested_cv.n_jobs}: {timer()}")
        nested_cv.eval()

    def _run_pretrain(self, train: FeatureSet):
        _save_path_pretrain = utils.check_make_dir(self.save_path.joinpath('pretrain_hps'), True)

        # always rewrite from cached bases
        self.temp_config.final_model.nested_cv_path = Path(
            str(self._nested_cv_base).replace('nested_cv', f'KFold_{self.ds_suffix}')
        )
        self.temp_config.final_model.load_path_feature_rankings = Path(
            str(self._full_rankings_base).replace("feat_rank/full", f"feat_rank/full/{self.ds_suffix}")
        )

        n_datasets = np.unique(train.dataset).shape[0]
        _save_postfix = 'Core' if n_datasets == 3 else 'Extended'

        model_manager_pretrain = ModelManager(self.temp_config, _save_path_pretrain)
        model_manager_pretrain.pretrain(train.copy(), dataset_save_tag=f'multiCenter{_save_postfix}')

    @staticmethod
    def _reconcile_and_update_config(settings_path: Path, updated_config: dict):
        """Compare config in settings.json with updated one and rewrite with _old/_updated keys if differences exist."""

        assert settings_path.is_file(), f"'{settings_path}' is not a file."
        settings = utils.read_from_json(settings_path)

        orig_config = settings.get("config", {})
        new_config = updated_config

        def __reconcile_dict(orig: dict, new: dict) -> dict:
            reconciled = orig.copy()
            for k, v_new in new.items():
                v_orig = orig.get(k)
                if isinstance(v_new, dict) and isinstance(v_orig, dict):
                    reconciled[k] = __reconcile_dict(v_orig, v_new)
                elif v_orig != v_new:
                    reconciled[f"{k}_old"] = v_orig
                    reconciled[f"{k}_updated"] = v_new
                else:
                    reconciled[k] = v_new
            return reconciled

        settings['config'] = __reconcile_dict(orig_config, new_config)
        utils.dump_to_json(settings, settings_path)
