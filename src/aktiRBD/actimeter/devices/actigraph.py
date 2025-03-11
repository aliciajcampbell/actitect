import logging
from pathlib import Path

import pandas as pd

from aktiRBD.actimeter.basedevice import BaseDevice

__all__ = ['Actigraph']

logger = logging.getLogger(__name__)


# todo: implement parsing function for GT3X and/or GT9X devices
# https://github.com/OxWearables/actipy/blob/master/src/actipy/ActigraphReader.java: adapt binary parsing to python (see Geneactive solution...)
# https://github.com/ghammad/pyActigraphy?

class Actigraph(BaseDevice):

    def __init__(self, path_to_cwa: Path, patient_id: str, kwargs: dict = None):
        raise NotImplementedError(f"'Actigraph' devices not supported yet.")
        super().__init__(filepath=path_to_cwa, patient_id=patient_id)
        self.kwargs = kwargs if kwargs \
            else {'include_gyro': False, 'include_temperature': False, 'include_light': False, 'include_mag': False}

    def __str__(self):
        return f"Actigraph(patient_ID='{self.meta['patient_id']}')"

    def _parse_binary_to_df(self):
        """ Parse the binary file and return it as pd.Dataframe.
        Parameters:
        Returns:
            :return: (pd.DataFrame) the raw data parsed as DataFrame.
        """
        logger.info(f"(io: {self.meta['patient_id']})"
                    f" loading from {self.processing_info['loading']['filepath']}")

        df = header = None
        raise NotImplementedError(f"'Actigraph' devices not supported yet.")
        return df, header
