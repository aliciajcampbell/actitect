import logging
from pathlib import Path
from typing import Union

from aktiRBD import utils
from aktiRBD.actimeter.devices import ActiGraph, AxivityAx6, GenericCSV, GENEActiv

__all__ = ['ActimeterFactory', 'SUPPORTED_FILETYPES']

logger = logging.getLogger(__name__)

EXT_2_DEVICE_MAP = {'.cwa': AxivityAx6, '.bin': GENEActiv, '.gt3x': ActiGraph, '.csv': GenericCSV}
DEVICE_KWARGS_MAP = {AxivityAx6: {'legacy_mode'}, GENEActiv: set(), ActiGraph: set(), GenericCSV: set()}
SUPPORTED_FILETYPES = sorted(list(EXT_2_DEVICE_MAP.keys()))


class ActimeterFactory:
    """ Entry point to all actimeter devices to automatically initialize the specific device based on file extension.
    Currently supported devices are:
        - Axivity Ax6: .cwa files
        - GENEActiv: .bin files
        - Actigraph: .gt3x files
        - GenericCSV: .csv files"""

    def __new__(cls, file_path: Union[str, Path], patient_id: str, mute: bool = False, **kwargs):
        """ Return an instance of the appropriate actimeter device subclass based on the file extension.
        Parameters:
            :param file_path: (Union[str, Path]) The path to the data file for the actimeter device.
            :param patient_id: (str) The patient identifier associated with the actimeter data.
            :param mute: (bool, Optional) If True, mutes the logging. Defaults to False.
        Returns:
            :return: (BaseDevice) instance of base device su corresponding to given actimeter type."""

        _ext = utils.get_file_extension(file_path)
        device_class = EXT_2_DEVICE_MAP.get(_ext)

        if device_class is None:
            raise ValueError(f" file extension '{_ext}' is not supported."
                             f" Available devices are: Axivity ('.cwa') and GENEActive ('.bin')")
        else:

            valid_keys = DEVICE_KWARGS_MAP.get(device_class, set())
            filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}

            if not mute:
                logger.info(f"(io: {patient_id}) detected '{device_class.__name__}' device.")
                ignored = set(kwargs) - valid_keys
                if ignored:
                    logger.warning(f"(io: {patient_id}) Ignored unsupported kwargs for"
                                   f" {device_class.__name__}: {ignored}")

            return device_class(file_path, patient_id, **filtered_kwargs)
