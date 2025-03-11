import logging
from pathlib import Path

from openmovement.load import CwaData
import pandas as pd

from aktiRBD.actimeter.basedevice import BaseDevice

__all__ = ['AxivityAx6']

logger = logging.getLogger(__name__)


# todo: implement binary parser myself to not rely on external package...

class AxivityAx6(BaseDevice):

    def __init__(self, path_to_cwa: Path, patient_id: str, kwargs: dict = None):
        super().__init__(filepath=path_to_cwa, patient_id=patient_id)
        self.kwargs = kwargs if kwargs \
            else {'include_gyro': False, 'include_temperature': False, 'include_light': False, 'include_mag': False}

    def __str__(self):
        return f"AxivityAx6(patient_ID='{self.meta['patient_id']}')"

    def _parse_binary_to_df(self, ax6_logging_threshold_d: int = 8):
        """ Parse the binary file and return it as pd.Dataframe.
        Parameters:
            :param ax6_logging_threshold_d: (int, Optional) If not None, use the 'LoggingStartTime' value from
                the binary header to trim the data. Useful if device was switched on to early and data has several
                days of non-wear before data taking. Only is applied if the data spans longer then
                'ax6_logging_threshold_d' days. The default is set to 8 but might be adjusted for longer data-taking.
        Returns:
            :return: (pd.DataFrame) the raw data parsed as DataFrame.
        """
        logger.info(f"(io: {self.meta['patient_id']})"
                    f" loading from {self.processing_info['loading']['filepath']}")
        with CwaData(self.processing_info['loading']['filepath'], include_time=True, include_accel=True, verbose=False,
                     **self.kwargs) as cwa_data:
            df = cwa_data.get_samples(use_datetime64=True)
            header = cwa_data.header

        df.rename(columns={'accel_x': 'x', 'accel_y': 'y', 'accel_z': 'z'}, inplace=True)
        df.set_index('time', inplace=True)
        header['sample_rate'] = header.pop('sampleRate')

        if ax6_logging_threshold_d and header.get('loggingStartTime'):
            logging_start_time = pd.to_datetime(header.get('loggingStartTime'), errors='coerce')
            if logging_start_time is not pd.NaT:
                data_span_days = (df.index[-1] - df.index[0]).total_seconds() / (24 * 60 * 60)
                if data_span_days > ax6_logging_threshold_d:  # trim it using 'loggingStartTime'
                    df = df[df.index >= logging_start_time]
                    logger.info(f"(io: {self.meta['patient_id']}) using 'loggingStartTime'"
                                f" ({pd.Timestamp(header['loggingStartTime'])}) to trim raw data.")
        else:
            if ax6_logging_threshold_d:
                logger.warning(f"(io: {self.meta['patient_id']}) 'loggingStartTime' not found in header;"
                               f" cannot apply logging start time cutoff.")

        return df, header
