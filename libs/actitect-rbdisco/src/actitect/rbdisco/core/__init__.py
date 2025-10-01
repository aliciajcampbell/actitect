from importlib import import_module
from typing import TYPE_CHECKING

__all__ = ['DataLoader', 'Evaluator', 'FeatureSet', 'Fold', 'ModelManager']

_LAZY_MAP = {
    'DataLoader': 'actitect.rbdisco.core.loader:DataLoader',
    'Evaluator': 'actitect.rbdisco.core.evaluator:Evaluator',
    'FeatureSet': 'actitect.rbdisco.core.types:FeatureSet',
    'Fold': 'actitect.rbdisco.core.types:Fold',
    'ModelManager': 'actitect.rbdisco.core.manager:ModelManager',
}


def __getattr__(name: str):
    try:
        target = _LAZY_MAP[name]
    except KeyError as e:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from e
    mod_path, attr = target.split(':', 1)
    mod = import_module(mod_path)
    return getattr(mod, attr)


def __dir__():
    return sorted(list(globals().keys()) + list(_LAZY_MAP.keys()))


if TYPE_CHECKING:  # for type-checkers only, no runtime
    from .loader import DataLoader
    from .evaluator import Evaluator
    from .types import FeatureSet, Fold
    from .manager import ModelManager
