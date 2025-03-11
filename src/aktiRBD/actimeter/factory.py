import logging
from pathlib import Path
from typing import Union

from aktiRBD import utils

from aktiRBD.actimeter.devices.axivity_ax6 import AxivityAx6
from aktiRBD.actimeter.devices.geneactiv import GENEActiv

__all__ = ['ActimeterFactory', 'SUPPORTED_FILETYPES']

logger = logging.getLogger(__name__)

EXT_2_DEVICE_MAP = {'.cwa': AxivityAx6, '.bin': GENEActiv, '.csv': NotImplementedError}
SUPPORTED_FILETYPES = sorted(list(EXT_2_DEVICE_MAP.keys()))


class ActimeterFactory:
    """  Entry point to all actimeter devices to automatically initialize the specific device based on file extension.

    Currently supported devices are :
        - Axivity Ax6: .cwa files
        - GENEActiv: .bin files """

    def __new__(cls, file_path: Union[str, Path], patient_id: str, mute: bool = False):
        """ Return an instance of the appropriate actimeter device subclass based on the file extension.

        Parameters:
            :param file_path: (Union[str, Path])  The path to the data file for the actimeter device.
            :param patient_id: (str) The patient identifier associated with the actimeter data.
            :param mute: (bool, Optional) If True, mutes the logging. Defaults to False.

        Returns:
            :return: (BaseDevice) instance of base device subclass corresponding to given actimeter type.
        """

        _ext = utils.get_file_extension(file_path)
        device_class = EXT_2_DEVICE_MAP.get(_ext)

        if device_class is None:
            raise ValueError(f" file extension '{_ext}' is not supported."
                             f" Available devices are: Axivity ('.cwa') and GENEActive ('.bin')")
        else:
            if not mute:
                logger.info(f"(io: {patient_id}) detected '{device_class.__name__}' device.")
            return device_class(file_path, patient_id)
