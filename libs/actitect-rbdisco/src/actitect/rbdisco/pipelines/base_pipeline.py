import logging
from abc import ABC, abstractmethod
from argparse import Namespace
from pathlib import Path
from typing import Union

from actitect.config import PipelineConfig, ExternalTestConfig
from ..core import DataLoader


logger = logging.getLogger(__name__)

__all__ = ['BasePipeline']


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
