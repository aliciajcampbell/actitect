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
                filepath = self.processing_info['loading']['filepath']

                delimiter = utils.detect_csv_delimiter(filepath)
                logger.debug(f'(io: {self.meta["patient_id"]}) detected delimiter: \'{delimiter}\'.')
                header_present, mapping = self._infer_csv_structure(filepath, delimiter)

                if header_present:
                    df = pd.read_csv(filepath, sep=delimiter, parse_dates=['time'])
                else:
                    df = pd.read_csv(
                        filepath, sep=delimiter, header=None, names=['time', 'x', 'y', 'z'], parse_dates=[0])

                df.rename(columns=str.lower, inplace=True)
                if mapping:
                    df.rename(columns=mapping, inplace=True)
                df.set_index('time', inplace=True)
                df = df[['x', 'y', 'z']]

                self.status_ok = 1
                return df, {'sample_rate': None}

            except Exception as e:
                logger.error("Exception occurred while parsing CSV: %s", e)
                self.status_ok = 0
                raise RuntimeError(f"CSV parsing failed: {e}") from e

    @staticmethod
    def _infer_csv_structure(filepath: Path, delimiter: str):
        """Inspect the first line to determine whether a header exists and
        map it to canonical columns (time,x,y,z) if possible."""
        from datetime import datetime

        with filepath.open('r', encoding='utf-8', errors='ignore') as f:
            first = f.readline().strip()

        tokens = [t.strip().lower() for t in first.split(delimiter)]

        # canonical header
        if tokens[:4] == ['time', 'x', 'y', 'z']:
            return True, None

        # detect headerless time,x,y,z
        try:
            datetime.fromisoformat(tokens[0])
            float(tokens[1])
            float(tokens[2])
            float(tokens[3])
            logger.warning('CSV has no header – assuming canonical format: time,x,y,z.')
            return False, None
        except Exception:
            pass

        logger.warning(f'CSV header not canonical ({tokens}). Attempting to map to canonical columns.')

        mapping = {}

        TIME_ALIASES = {'time', 'timestamp', 'datetime', 'date'}
        X_ALIASES = {'x', 'acc_x', 'ax', 'accel_x', 'a_x'}
        Y_ALIASES = {'y', 'acc_y', 'ay', 'accel_y', 'a_y'}
        Z_ALIASES = {'z', 'acc_z', 'az', 'accel_z', 'a_z'}

        for t in tokens:
            if t in TIME_ALIASES:
                mapping[t] = 'time'
            elif t in X_ALIASES:
                mapping[t] = 'x'
            elif t in Y_ALIASES:
                mapping[t] = 'y'
            elif t in Z_ALIASES:
                mapping[t] = 'z'

        required = {'time', 'x', 'y', 'z'}

        if set(mapping.values()) != required:
            raise RuntimeError(f'Unable to map CSV header {tokens} to canonical columns: time,x,y,z.')

        return True, mapping
