from pathlib import Path
from dataclasses import dataclass, fields, asdict
from typing import Type, TypeVar, Any, Dict, List, Union

import yaml

__all__ = ['PipelineConfig', 'ModelConfig', 'DataConfig', 'NestedCVConfig',
           'FinalModelConfig', 'ExternalTestConfig', 'ExperimentConfig', 'LoaderConfig']


@dataclass
class BaseConfig:
    """Base dataclass to load pipeline.yaml config file."""
    T = TypeVar('T', bound='BaseConfig')

    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        init_kwargs = {}
        for field_ in fields(cls):
            field_name = field_.name
            field_type = field_.type
            if field_name not in data:
                raise KeyError(f"Missing key '{field_name}' in configuration.")
            value = data[field_name]

            # Handle nested dataclasses
            if hasattr(field_type, '__dataclass_fields__'):
                init_kwargs[field_name] = field_type.from_dict(value)

            # Handle lists of dataclasses
            elif getattr(field_type, '__origin__', None) is list:
                elem_type = field_type.__args__[0]  # Extract list element type
                if hasattr(elem_type, '__dataclass_fields__'):  # Ensure elements are dataclasses
                    init_kwargs[field_name] = [elem_type.from_dict(v) for v in value]
                else:
                    init_kwargs[field_name] = value  # Handle regular lists (e.g., List[str])

            else:
                init_kwargs[field_name] = value

        return cls(**init_kwargs)

    @classmethod
    def from_yaml(cls, yaml_path: Path = None) -> 'BaseConfig':
        if not yaml_path:
            yaml_path = Path(__file__).parent.joinpath('pipeline.yaml')

        with yaml_path.open('r') as file:
            config_dict = yaml.safe_load(file)
        return cls.from_dict(config_dict)

    def dict(self) -> dict:
        return asdict(self)

    def copy(self: T) -> T:
        return type(self).from_dict(self.dict())


@dataclass
class EarlyStoppingConfig(BaseConfig):
    early_stopping_rounds: int
    eval_metric: List[str]


@dataclass
class BayesParamsConfig(BaseConfig):
    n_calls: int
    use_early_stopping: bool
    verbose: bool


@dataclass
class ModelConfig(BaseConfig):
    which: str
    dummy: str
    early_stopping: EarlyStoppingConfig
    bayes_params: BayesParamsConfig


@dataclass
class ProcessingConfig(BaseConfig):
    scaling_order: str
    use_smote: bool
    scaler: str = None  # 'minmax', 'standard', 'robust' or None


@dataclass
class LoaderConfig(BaseConfig):
    binary_mapping: dict
    feature_dir: str  # relative to patient_dir
    aggregation: List[str]
    included_local_features: List[str]
    included_global_features: List[str]


@dataclass
class DataConfig(BaseConfig):
    agg_level: str
    processing: ProcessingConfig
    loader: LoaderConfig


@dataclass
class CVConfig(BaseConfig):  # needs to be compatible with sklearn.model_selection.BaseCrossValidator kwargs
    n_splits: int
    shuffle: bool


@dataclass
class ExperimentConfig(BaseConfig):
    name: str
    threshold: str
    patient_aggregation: str
    hp_setup_name: str = None
    early_stopping: bool = None
    calibration: str = None


@dataclass
class NestedCVConfig(BaseConfig):
    n_repeats: int
    n_jobs: int
    outer_cv: CVConfig
    inner_cv: CVConfig
    n_interp_points_roc_pr: int
    n_interp_points_calibration: int
    calib_plot_hist_bin_width: float
    default_experiment: ExperimentConfig
    log_night_eval: bool
    stratify_by_dataset_if_pooled: bool
    load_path_cv_feature_rankings: Union[str, Path] = None


@dataclass
class FinalModelConfig(BaseConfig):
    n_jobs: int
    bayes_cv: CVConfig
    bayes_params: BayesParamsConfig
    stratify_by_dataset_if_pooled: bool
    early_stopping: EarlyStoppingConfig
    include_pretrain_merged: bool
    overwrite_final_repo_models: bool
    log_night_level: bool
    output_patient_csv: bool
    experiments: List[ExperimentConfig]
    load_path_feature_rankings: Union[str, Path] = None
    nested_cv_path: Union[str, Path] = None
    save_path_models: Union[str, Path, list] = None
    calibration_cv: CVConfig = None


@dataclass
class PipelineConfig(BaseConfig):
    random_state: int
    model: ModelConfig
    data: DataConfig
    nested_cv: NestedCVConfig
    final_model: FinalModelConfig


@dataclass
class ExternalTestConfig(BaseConfig):
    random_state: int
    n_jobs: int
    pretrained_model_dirs: List[str]
    data: DataConfig
    log_night_eval: bool
    output_patient_csv: bool
