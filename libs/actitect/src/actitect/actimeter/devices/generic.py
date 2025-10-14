import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from ... import utils
from ..basedevice import BaseDevice

__all__ = ['Generic']

logger = logging.getLogger(__name__)


class Generic(BaseDevice):
    """ Subclass of BaseDevice to load data from device-agnostic CSV files or a preloaded pandas.DataFrame.
     The csv file/Dataframe must have a header with columns 'time' and 'x', 'y', 'z'. The 'time' column should be
      formatted according to ISO 8601 (%Y-%m-%dT%H:%M:%S.%f%z) and the data columns should be numeric in units of g."""

    def __init__(
            self,
            filepath: Optional[Path],
            patient_id: str,
            *,
            raw_df: Optional[pd.DataFrame] = None,
            header: Optional[dict] = None,
    ):
        super().__init__(filepath=filepath, patient_id=patient_id, raw_df=raw_df, header=header)

    def __str__(self):
        return f"Generic(patient_ID='{self.meta['patient_id']}')"

    def _parse_binary_to_df(self, resolve_duplicates: bool = True, header_only: bool = False):
        if header_only:
            raise ValueError("'header_only' is not supported for Generic device.")
        else:

            try:
                delimiter = utils.detect_csv_delimiter(self.processing_info['loading']['filepath'])
                logger.debug(f"(io: {self.meta['patient_id']}) detected delimiter: '{delimiter}'.")

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
