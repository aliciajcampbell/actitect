from dataclasses import dataclass
from typing import Any, Dict, List, Union, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from skopt.space import Real, Categorical, Integer

from aktiRBD.config import TopKFeatsConfig

__all__ = ['model_factory', 'ModelSetup', 'space_to_bounds']


@dataclass
class ModelSetup:
    name: str
    model: Any
    default_params: Dict[str, Any]
    bayesian_param_space: List[Union[Integer, Real]]
    bayesian_fixed_params: Dict[str, Any]


def model_factory(model_name: str, cls_balance: float, seed: int,
                  top_k_cfg: Optional[TopKFeatsConfig] = None) -> ModelSetup:

    if top_k_cfg is not None:
        topk_dim, fixed_topk = build_topk_dimension(top_k_cfg.dict())
        common_bayes_param_space = [topk_dim] if topk_dim is not None else []
        common_fixed = fixed_topk
    else:
        common_bayes_param_space = []
        common_fixed = {}

    if model_name == 'xgboost':
        return ModelSetup(
            name=model_name,
            model=XGBClassifier,
            default_params={
                'random_state': seed,

                # general
                'booster': 'gbtree',
                # which booster: 'gbtree'(default)/'dart' for trees, 'gblinear' for linear functions
                'verbosity': 1,  # 0:silent, 1: warn, 2: info, 3: debug

                # column subsampling -> how many features are rand. selected during tree construction (a bit like
                # dropout)
                'colsample_bylevel': .5,  # frac of feat. at each level of tree
                'colsample_bytree': 1.0,  # frac of feat. for each tree
                'colsample_bynode': 1.0,  # frac of feat at each split/node

                'n_estimators': 100,

                # tree booster (gbtree/dart):
                'max_depth': 5,
                # default: 6, max depth of single tree, increasing: more complex  (trade-off) (0=no-limit)
                'min_child_weight': 1,
                # default: 1, min N of samples/instance weights needed to form new child node
                # Low (e.g. 1,2,3) -> trees split even with rel. small N of samples -> high complexity (tradeoff)
                # High (e.g 5 and above) -> simpler trees, might help overfitting but can prevent learning

                'learning_rate': .1,
                # default: .3, step size shrinkage of features after each boosting step. So scales down the
                # impact of each new tree to ensemble, preventing it from over-fitting too quickly.
                # note: increase num_round when decreasing eta!
                # Low (e.g. 0.001 - 0.1) -> slower learning, more robust  (would need more n_estimators to converge!)
                # High (e.g 0.3-0.5 and above) -> faster learning, but can lead to over-fitting

                'gamma': 2,
                # default 0, minimum reduction in loss function for a split to provide for being added as a node.
                # Low (e.g. 0-1) -> more frequent splitting, increases  complexity (capability vs overfitting)
                # High ( > 1) -> reduces complexity, can help overfitting but can also make  not complex enough

                'reg_alpha': .5,  # L1, increasing -> more conservative, range 0, inf
                'reg_lambda': .5,  # L1, increasing -> more conservative, range 0, inf

                'subsample': 1,
                # default: 1, like dropout, rows(samples) and not columns(feats) like colsample, range: (0,1]
                'sampling_method': 'uniform',  # 'uniform'(default) or 'gradient_based': how to sample if subsample != 1

                'tree_method': 'auto',  # 'auto'(def.), 'exact', 'approx', 'hist': alg. to construct trees

                'scale_pos_weight': cls_balance,
                # default: 1, control balance of pos./neg. weights for unbal. data ref: n_neg/n_pos
            },
            bayesian_param_space=common_bayes_param_space + [
                # TODO: reduce DOF
                Integer(name='n_estimators', low=10, high=1_000, prior='uniform'),
                Integer(name='max_depth', low=1, high=14, prior='uniform'),
                Real(name='reg_alpha', low=0, high=10, prior='uniform'),
                Real(name='reg_lambda', low=0, high=10, prior='uniform'),
                Real(name='colsample_bylevel', low=0.05, high=1.0, prior='uniform'),
                Real(name='colsample_bytree', low=0.05, high=1.0, prior='uniform'),
                Real(name='learning_rate', low=1e-3, high=1, prior='log-uniform'),
                Real(name='min_child_weight', low=1, high=100, prior='log-uniform'),
                Real(name='subsample', low=0.1, high=0.99, prior='uniform')
            ],
            bayesian_fixed_params={
                **common_fixed,
                'random_state': seed,
                'booster': 'gbtree',
                # which booster: 'gbtree'(default)/'dart' for trees, 'gblinear' for linear functions
                'verbosity': 1,  # 0:silent, 1: warn, 2: info, 3: debug
                'colsample_bytree': 1.0,  # frac of feat. for each tree
                'colsample_bynode': 1.0,  # frac of feat at each split/node
                'sampling_method': 'uniform',  # 'uniform'(default) or 'gradient_based': how to sample if subsample != 1
                'tree_method': 'auto',  # 'auto'(def.), 'exact', 'approx', 'hist': alg. to construct trees
                'scale_pos_weight': cls_balance,
                'max_leaves': None,
                'grow_policy': None,
                # 0: favor splitting at nodes closest to node, i.e. grow depth-wise, 1 at nodes with highest loss change
                'gamma': None,
                # min loss reduction required to make further partition on a leaf node ('min_split_loss')
                'validate_parameters': True,
                'max_delta_step': None,
            },
        )

    elif model_name == 'svm':
        return ModelSetup(
            name=model_name,
            model=SVC,
            default_params={
                'C': 1.0,  # Regularization parameter, smaller: smoother dec. surface, larger: more complex dec. surface
                'kernel': 'rbf',  # Kernel type: 'linear', 'poly', 'rbf', 'sigmoid'
                'gamma': 'scale',  # Kernel coefficient: 'scale' or 'auto'
                'degree': 3,  # degree of polynomial in case of 'poly'
                'random_state': seed,
                'probability': True,  # To enable probability estimates
                'class_weight': 'balanced',
            },
            bayesian_param_space=common_bayes_param_space + [
                Real(name='C', low=1e-6, high=1e+6, prior='log-uniform'),
                Real(name='gamma', low=1e-6, high=1e+1, prior='log-uniform'),
                Categorical(name='kernel', categories=['linear', 'poly', 'rbf', 'sigmoid']),
                Integer(name='degree', low=2, high=5, prior='uniform')  # Used only with 'poly' kernel
            ],
            bayesian_fixed_params={
                **common_fixed,
                'random_state': seed,
                'probability': True,  # To enable probability estimates
                'class_weight': 'balanced'
            },
        )

    elif model_name == 'rf':
        return ModelSetup(
            name=model_name,
            model=RandomForestClassifier,
            default_params={
                'n_estimators': 100,  # Number of trees in the forest
                'max_depth': None,  # Maximum depth of the tree; None means nodes are expanded until all leaves are pure
                'min_samples_split': 2,  # Minimum number of samples required to split an internal node
                'min_samples_leaf': 1,  # Minimum number of samples required to be at a leaf node
                'max_features': 'sqrt',  # Number of features to consider when looking for the best split
                'bootstrap': True,  # Whether bootstrap samples are used when building trees
                'random_state': seed,  # Seed for reproducibility
                'class_weight': 'balanced',  # Automatically adjust weights inversely proportional to class frequencies
            },
            bayesian_param_space=common_bayes_param_space + [
                Integer(name='n_estimators', low=50, high=1000, prior='uniform'),
                Integer(name='max_depth', low=1, high=50, prior='uniform'),
                Integer(name='min_samples_split', low=2, high=20, prior='uniform'),
                Integer(name='min_samples_leaf', low=1, high=20, prior='uniform'),
                Real(name='max_features', low=0.1, high=1.0, prior='uniform'),  # Proportion of features to consider
                Categorical(name='bootstrap', categories=[True, False])
            ],
            bayesian_fixed_params={
                **common_fixed,
                'random_state': seed,
                'class_weight': 'balanced',
            },
        )

    elif model_name == 'lr':
        return ModelSetup(
            name=model_name,
            model=LogisticRegression,
            default_params={
                'penalty': 'l2',
                'C': 1.0,
                'solver': 'lbfgs',
                'max_iter': 100,
                'class_weight': None,
                'multi_class': 'auto',
                'random_state': seed
            },
            bayesian_param_space=common_bayes_param_space + [
                Categorical(name='penalty', categories=['l2', 'l1', 'elasticnet', 'none']),
                Real(name='C', low=1e-6, high=1e+6, prior='log-uniform'),
                Categorical(name='solver', categories=['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']),
                Integer(name='max_iter', low=50, high=500, prior='uniform'),
                Categorical(name='class_weight', categories=[None, 'balanced']),
                Categorical(name='multi_class', categories=['auto', 'ovr', 'multinomial']),
            ],
            bayesian_fixed_params={
                **common_fixed,
                'random_state': seed,
            },
        )

    elif model_name == 'knn':
        return ModelSetup(
            name=model_name,
            model=KNeighborsClassifier,
            default_params={
                'n_neighbors': 5,
                'weights': 'uniform',
                'algorithm': 'auto',
                'leaf_size': 30,
                'p': 2,
                'metric': 'minkowski',
            },
            bayesian_param_space=common_bayes_param_space + [
                Integer(name='n_neighbors', low=1, high=30, prior='uniform'),
                Categorical(name='weights', categories=['uniform', 'distance']),
                Categorical(name='algorithm', categories=['auto', 'ball_tree', 'kd_tree', 'brute']),
                Integer(name='leaf_size', low=10, high=100, prior='uniform'),
                Integer(name='p', low=1, high=5, prior='uniform'),
                Categorical(name='metric', categories=['minkowski', 'euclidean', 'manhattan']),
            ],
            bayesian_fixed_params={
                **common_fixed,
                'n_jobs': -1,  # Using all processors for distance computation if available
            },
        )

    elif model_name == 'light_gbm':
        return ModelSetup(
            name=model_name,
            model=NotImplementedError,
            default_params={'todo': 'todo'},
            bayesian_param_space={'todo': 'todo'},
            bayesian_fixed_params={'todo': 'todo'}, )

    else:
        raise ValueError(f"Model {model_name} is not supported.")


def build_topk_dimension(top_k_cfg: dict):
    """Returns:
        space_dim – Integer(...) or None
        fixed – dict {'top_k_feats': value} or {}"""
    if top_k_cfg.get("tune", True):
        return Integer(
            name="top_k_feats",
            low=int(top_k_cfg["low"]),
            high=int(top_k_cfg["high"]),
            prior="uniform"
        ), {}
    else:
        return None, {"top_k_feats": int(top_k_cfg["fixed_value"])}


def space_to_bounds(space) -> dict:
    """
    Parameters
    ----------
    space : list[Dimension]
        e.g. model_setup.bayesian_param_space — any mix of
        skopt.space.Integer / Real / Categorical.

    Returns
    -------
    dict
        {<dimension name>: (low, high)}  — ready for `hp_stability(..)`
        For categoricals the index range (0 … n-1) is returned.
    """
    bounds = {}
    for i, dim in enumerate(space):
        # make sure every dimension has a usable name
        name = dim.name or f"dim_{i}"

        if isinstance(dim, (Integer, Real)):
            bounds[name] = (float(dim.low), float(dim.high))

        elif isinstance(dim, Categorical):
            bounds[name] = (0, len(dim.categories) - 1)

        else:  # pragma: no cover
            raise TypeError(f"Unsupported skopt dimension type: {type(dim)}")

    return bounds
