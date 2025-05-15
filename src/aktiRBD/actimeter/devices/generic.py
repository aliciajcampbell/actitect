import logging
from pathlib import Path

import pandas as pd

from aktiRBD import utils
from aktiRBD.actimeter.basedevice import BaseDevice

__all__ = ['GenericCSV']

logger = logging.getLogger(__name__)


class GenericCSV(BaseDevice):
    """ Subclass of BaseDevice to load data from device-agnostic CSV files. The csv file must have a header with columns
     'time' and 'x', 'y', 'z'. The 'time' column should be formatted according to ISO 8601 (%Y-%m-%dT%H:%M:%S.%f%z) and
     the data columns should be numeric in units of g."""

    def __init__(self, path_to_csv: Path, patient_id: str):
        super().__init__(filepath=path_to_csv, patient_id=patient_id)

    def __str__(self):
        return f"GenericCSV(patient_ID='{self.meta['patient_id']}')"

    def _parse_binary_to_df(self, resolve_duplicates: bool = True, header_only: bool = False):
        logger.info(f"(io: {self.meta['patient_id']}) loading from '{self.processing_info['loading']['filepath']}'.")
        if header_only:
            raise ValueError("'header_only' is not supported for GenericCSV device.")
        else:

            try:
                delimiter = utils.detect_csv_delimiter(self.processing_info['loading']['filepath'])
                logger.info(f"(io: {self.meta['patient_id']}) detected delimiter: '{delimiter}'.")

                df = pd.read_csv(self.processing_info['loading']['filepath'], sep=delimiter, parse_dates=['time'])
                df.rename(columns=str.lower, inplace=True)  # ensure x/y/z columns are lowercase
                df.set_index('time', inplace=True)
                df = df[['x', 'y', 'z']]  # keep only required cols

                self.status_ok = 1
                return df, {'sample_rate': None}

            except Exception as e:
                logger.error(f"Exception occurred: {e}")
                self.status_ok = 0
                return None
