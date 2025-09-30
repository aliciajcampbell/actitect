from dataclasses import dataclass
import logging
from typing import Any, Dict, List, Optional, Union, Tuple

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from skopt.space import Real, Categorical, Integer
from xgboost import XGBClassifier

from actitect.config import TopKFeatsConfig

logger = logging.getLogger(__name__)

__all__ = ["ModelFactory", "ModelSetup", "space_to_bounds"]


@dataclass
class ModelSetup:
    name: str
    model: Any
    default_params: Dict[str, Any]
    bayesian_param_space: List[Union[Integer, Real, Categorical]]
    bayesian_fixed_params: Dict[str, Any]


class ModelFactory:
    """
    Class-based factory with:
      - TopKFeatsConfig support (tune or fix top_k_feats)
      - Stage overrides (fix/clamp/restrict search dims)
      - space_to_bounds() helper retained for HP stability tools
    """

    def __init__(
            self,
            model_name: str,
            cls_balance: float,
            seed: int,
            top_k_cfg: Optional[TopKFeatsConfig] = None,
            overrides: Optional[Dict[str, Any]] = None,
    ):
        self.model_name = model_name
        self.cls_balance = cls_balance
        self.seed = seed
        self.overrides = overrides or {}

        # Build the optional top-k search dimension (or fixed) from config
        if top_k_cfg is not None:
            topk_dim, fixed_topk = self._build_topk_dimension(top_k_cfg.dict())
            self.common_bayes_param_space: List[Union[Integer, Real, Categorical]] = (
                [topk_dim] if topk_dim is not None else []
            )
            self.common_fixed = fixed_topk  # e.g. {'top_k_feats': 32} if fixed
        else:
            self.common_bayes_param_space = []
            self.common_fixed = {}

    def build(self) -> ModelSetup:
        setup = self._get_setup()
        # Apply stage overrides (if any): move fixed values to fixed_params, clamp numeric, restrict categoricals
        if self.overrides:
            new_space, fixed = self._apply_stage_overrides(setup.bayesian_param_space, self.overrides)

            # Log overrides in a human-readable way
            msg_lines = ["Applied HP overrides:"]
            for hp, override in self.overrides.items():
                old_val = self._dim_lookup(setup.bayesian_param_space, setup.bayesian_fixed_params, hp)
                msg_lines.append(f"  {hp:20} | old: {old_val:<25} → new: {override}")
            logger.warning("\n" + "\n".join(msg_lines))

            setup.bayesian_param_space = new_space
            setup.bayesian_fixed_params.update(fixed)

        return setup

    def _get_setup(self) -> ModelSetup:
        if self.model_name == "xgboost":
            return ModelSetup(
                name=self.model_name,
                model=XGBClassifier,
                default_params={
                    "random_state": self.seed,
                    # general
                    "booster": "gbtree",
                    "verbosity": 1,  # 0:silent, 1: warn, 2: info, 3: debug
                    # column subsampling
                    "colsample_bylevel": 0.5,
                    "colsample_bytree": 1.0,
                    "colsample_bynode": 1.0,
                    # base learners
                    "n_estimators": 100,
                    "max_depth": 5,
                    "min_child_weight": 1,
                    "learning_rate": 0.1,
                    "gamma": 2,
                    "reg_alpha": 0.5,
                    "reg_lambda": 0.5,
                    "subsample": 1.0,
                    "sampling_method": "uniform",
                    "tree_method": "auto",
                    "scale_pos_weight": self.cls_balance,
                },
                bayesian_param_space=self.common_bayes_param_space
                                     + [
                                         Integer(name="n_estimators", low=10, high=1_000, prior="uniform"),
                                         Integer(name="max_depth", low=1, high=14, prior="uniform"),
                                         Real(name="reg_alpha", low=0, high=10, prior="uniform"),
                                         Real(name="reg_lambda", low=0, high=10, prior="uniform"),
                                         Real(name="colsample_bylevel", low=0.05, high=1.0, prior="uniform"),
                                         Real(name="colsample_bytree", low=0.05, high=1.0, prior="uniform"),
                                         Real(name="learning_rate", low=1e-3, high=1, prior="log-uniform"),
                                         Real(name="min_child_weight", low=1, high=100, prior="log-uniform"),
                                         Real(name="subsample", low=0.1, high=0.99, prior="uniform"),
                                     ],
                bayesian_fixed_params={
                    **self.common_fixed,  # include fixed top_k_feats if set by TopKFeatsConfig
                    "random_state": self.seed,
                    "booster": "gbtree",
                    "verbosity": 1,
                    "colsample_bytree": 1.0,
                    "colsample_bynode": 1.0,
                    "sampling_method": "uniform",
                    "tree_method": "auto",
                    "scale_pos_weight": self.cls_balance,
                    "max_leaves": None,
                    "grow_policy": None,
                    "gamma": None,  # min_split_loss
                    "validate_parameters": True,
                    "max_delta_step": None,
                },
            )

        elif self.model_name == "svm":
            return ModelSetup(
                name=self.model_name,
                model=SVC,
                default_params={
                    "C": 1.0,
                    "kernel": "rbf",
                    "gamma": "scale",
                    "degree": 3,
                    "random_state": self.seed,
                    "probability": True,
                    "class_weight": "balanced",
                },
                bayesian_param_space=self.common_bayes_param_space
                                     + [
                                         Real(name="C", low=1e-6, high=1e6, prior="log-uniform"),
                                         Real(name="gamma", low=1e-6, high=1e1, prior="log-uniform"),
                                         Categorical(name="kernel", categories=["linear", "poly", "rbf", "sigmoid"]),
                                         Integer(name="degree", low=2, high=5, prior="uniform"),
                                     ],
                bayesian_fixed_params={
                    **self.common_fixed,
                    "random_state": self.seed,
                    "probability": True,
                    "class_weight": "balanced",
                },
            )

        elif self.model_name == "rf":
            return ModelSetup(
                name=self.model_name,
                model=RandomForestClassifier,
                default_params={
                    "n_estimators": 100,
                    "max_depth": None,
                    "min_samples_split": 2,
                    "min_samples_leaf": 1,
                    "max_features": "sqrt",
                    "bootstrap": True,
                    "random_state": self.seed,
                    "class_weight": "balanced",
                },
                bayesian_param_space=self.common_bayes_param_space
                                     + [
                                         Integer(name="n_estimators", low=50, high=1000, prior="uniform"),
                                         Integer(name="max_depth", low=1, high=50, prior="uniform"),
                                         Integer(name="min_samples_split", low=2, high=20, prior="uniform"),
                                         Integer(name="min_samples_leaf", low=1, high=20, prior="uniform"),
                                         Real(name="max_features", low=0.1, high=1.0, prior="uniform"),
                                         Categorical(name="bootstrap", categories=[True, False]),
                                     ],
                bayesian_fixed_params={
                    **self.common_fixed,
                    "random_state": self.seed,
                    "class_weight": "balanced",
                },
            )

        elif self.model_name == "lr":
            return ModelSetup(
                name=self.model_name,
                model=LogisticRegression,
                default_params={
                    "penalty": "l2",
                    "C": 1.0,
                    "solver": "lbfgs",
                    "max_iter": 100,
                    "class_weight": None,
                    "multi_class": "auto",
                    "random_state": self.seed,
                },
                bayesian_param_space=self.common_bayes_param_space
                                     + [
                                         Categorical(name="penalty", categories=["l2", "l1", "elasticnet", "none"]),
                                         Real(name="C", low=1e-6, high=1e6, prior="log-uniform"),
                                         Categorical(name="solver",
                                                     categories=["newton-cg", "lbfgs", "liblinear", "sag", "saga"]),
                                         Integer(name="max_iter", low=50, high=500, prior="uniform"),
                                         Categorical(name="class_weight", categories=[None, "balanced"]),
                                         Categorical(name="multi_class", categories=["auto", "ovr", "multinomial"]),
                                     ],
                bayesian_fixed_params={
                    **self.common_fixed,
                    "random_state": self.seed,
                },
            )

        elif self.model_name == "knn":
            return ModelSetup(
                name=self.model_name,
                model=KNeighborsClassifier,
                default_params={
                    "n_neighbors": 5,
                    "weights": "uniform",
                    "algorithm": "auto",
                    "leaf_size": 30,
                    "p": 2,
                    "metric": "minkowski",
                },
                bayesian_param_space=self.common_bayes_param_space
                                     + [
                                         Integer(name="n_neighbors", low=1, high=30, prior="uniform"),
                                         Categorical(name="weights", categories=["uniform", "distance"]),
                                         Categorical(name="algorithm",
                                                     categories=["auto", "ball_tree", "kd_tree", "brute"]),
                                         Integer(name="leaf_size", low=10, high=100, prior="uniform"),
                                         Integer(name="p", low=1, high=5, prior="uniform"),
                                         Categorical(name="metric", categories=["minkowski", "euclidean", "manhattan"]),
                                     ],
                bayesian_fixed_params={
                    **self.common_fixed,
                    "n_jobs": -1,
                },
            )

        elif self.model_name == "light_gbm":
            return ModelSetup(
                name=self.model_name,
                model=NotImplementedError,
                default_params={"todo": "todo"},
                bayesian_param_space={"todo": "todo"},
                bayesian_fixed_params={"todo": "todo"},
            )

        else:
            raise ValueError(f"Model {self.model_name} is not supported.")

    @staticmethod
    def _build_topk_dimension(top_k_cfg: Dict[str, Any]):
        if top_k_cfg.get('tune', True):
            return Integer(
                name='top_k_feats', low=int(top_k_cfg['low']), high=int(top_k_cfg['high']), prior='uniform'), {}
        else:
            return None, dict(top_k_feats=int(top_k_cfg['fixed_value']))

    @staticmethod
    def _dim_lookup(space, fixed, hp):
        for dim in space:
            if dim.name == hp:
                if isinstance(dim, (Integer, Real)):
                    return f"[{dim.low}, {dim.high}]"
                if isinstance(dim, Categorical):
                    return list(dim.categories)
        return fixed.get(hp, "<not set>")

    @staticmethod
    def _apply_stage_overrides(base_space: List[Union[Integer, Real, Categorical]], overrides: Dict[str, Any]):
        """ overrides: {hp_name: scalar | [low, high] | [cat1, cat2, ...]}
          - scalar -> fixed param (removed from search space)
          - [low, high] for numeric -> clamp within original bounds, keep prior
          - [choices...] for categorical -> restrict; len==1 -> fixed"""
        name_to_dim = {d.name: d for d in base_space}
        new_space: List[Union[Integer, Real, Categorical]] = []
        fixed_params: Dict[str, Any] = {}

        # Warn about keys not present in the original space
        for key in overrides.keys():
            if key not in name_to_dim:
                print(f"[ModelFactory] WARNING: override '{key}' not in base search space; ignoring.")

        for dim in base_space:
            name = dim.name
            if name not in overrides:
                new_space.append(dim)
                continue

            ov = overrides[name]

            # Case 1: fixed scalar
            if isinstance(ov, (int, float, str)):
                fixed_params[name] = ov
                continue

            # Case 2: categorical restriction
            if isinstance(dim, Categorical) and isinstance(ov, list):
                if len(ov) == 0:
                    raise ValueError(f"Empty list override for categorical '{name}'.")
                base_choices = list(dim.categories)
                for v in ov:
                    if v not in base_choices:
                        raise ValueError(
                            f"Override value '{v}' not in categorical '{name}' choices {base_choices}"
                        )
                if len(ov) == 1:
                    fixed_params[name] = ov[0]
                else:
                    new_space.append(Categorical(ov, name=name))
                continue

            # Case 3: numeric clamp [low, high]
            if isinstance(ov, list) and len(ov) == 2 and not isinstance(dim, Categorical):
                low, high = ov
                if low > high:
                    raise ValueError(f"Override low>high for '{name}': {ov}")

                if isinstance(dim, Integer):
                    orig_low, orig_high = dim.low, dim.high
                    low = max(orig_low, int(low))
                    high = min(orig_high, int(high))
                    if low > high:
                        raise ValueError(
                            f"Override for '{name}' outside original bounds [{orig_low}, {orig_high}]."
                        )
                    new_space.append(Integer(low=low, high=high, prior=dim.prior, name=name))
                elif isinstance(dim, Real):
                    orig_low, orig_high = dim.low, dim.high
                    low = max(orig_low, float(low))
                    high = min(orig_high, float(high))
                    if low > high:
                        raise ValueError(
                            f"Override for '{name}' outside original bounds [{orig_low}, {orig_high}]."
                        )
                    new_space.append(Real(low=low, high=high, prior=dim.prior, name=name))
                else:
                    raise ValueError(f"Unexpected dim type for '{name}'")
                continue

            if isinstance(ov, list):
                # List provided but not categorical & not [low, high]
                raise ValueError(f"Override format not understood for '{name}': {ov}")

            raise ValueError(f"Unsupported override type for '{name}': {type(ov)}")

        return new_space, fixed_params


def space_to_bounds(space: List[Union[Integer, Real, Categorical]]) -> Dict[str, Tuple[float, float]]:
    """ Convert a skopt space into {name: (low, high)} bounds.
    For categoricals, returns (0, n-1) — useful for stability plots/normalization."""
    bounds: Dict[str, Tuple[float, float]] = {}
    for i, dim in enumerate(space):
        name = dim.name or f"dim_{i}"
        if isinstance(dim, (Integer, Real)):
            bounds[name] = (float(dim.low), float(dim.high))
        elif isinstance(dim, Categorical):
            bounds[name] = (0.0, float(len(dim.categories) - 1))
        else:  # pragma: no cover
            raise TypeError(f"Unsupported skopt dimension type: {type(dim)}")
    return bounds
