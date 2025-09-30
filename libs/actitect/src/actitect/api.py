# todo: this should be the ActTect Programmable API, i.e. users should be able to do things like
# todo: make a seperate README for the programmable API
# from actitect.api import ActimeterFactory
# actimeter = ActimeterFactory(..)
# data_df = actimeter.load_raw_data()
# processed_df = actimeter.process()
from pathlib import Path
from importlib.metadata import version as _pkg_version

__version__ = _pkg_version("actitect")

# re-export the common device classes (lightweight imports only)
from .actimeter import ActimeterFactory, AxivityAx6, ActiGraph, GENEActiv, GenericCSV


__all__ = [
    '__version__',
    'ActimeterFactory',
    'AxivityAx6',
    'ActiGraph',
    'GENEActiv',
    'GenericCSV',
]



# todo: define some wrapeprs to e.g. just load the data from a given file, process it with some options, e.g. preprocessing and control over segmentation, a plotting function etc.
def process_file(file_path: Path):
    device = ActimeterFactory()
    raise NotImplementedError