import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from zipfile import ZipFile

import numpy as np
import pandas as pd

from ..basedevice import BaseDevice

from ..settings import ActiGraphLogRecordV1, ActiGraphLogRecordV2

__all__ = ['ActiGraph']

logger = logging.getLogger(__name__)


class ActiGraph(BaseDevice):  # noqa

    def __init__(self, path_to_gt3x: Path, patient_id: str, kwargs: dict = None):
        super().__init__(filepath=path_to_gt3x, patient_id=patient_id)

        logger.warning("[NOQA]: ActiGraph parser has not been fully validated yet. Use with caution.")

        self.zip_file = ZipFile(path_to_gt3x, 'r')
        self.device_header = None

        if self.is_in_nhanes_format:
            self.LogRecord = ActiGraphLogRecordV1
            raise NotImplementedError(  # todo: eventually add NHANES support if needed
                "Detected device is in deprecated v1 (NHANES) format, i.e., does not contain the 'log.bin' file. "
                "Support may be added in the future.")
        else:
            self.LogRecord = ActiGraphLogRecordV2

        self._assert_ism_was_disabled()

        self.kwargs = kwargs if kwargs else {'incl_light': False, 'incl_battery': False}
        self.device_header, self.sample_rate = None, None

    def __str__(self):
        return f"ActiGraph(patient_ID='{self.meta['patient_id']}')"

    def _parse_binary_to_df(self, header_only: bool = False):

        try:
            # get header from 'info.txt'
            self.device_header, self.sample_rate = self._load_g3tx_info()
            if header_only:
                return pd.DataFrame(), self.device_header

            record_buffers = {'ACTIVITY': [], 'ACTIVITY2': []}
            if self.kwargs.get('incl_battery', False):
                record_buffers['BATTERY'] = []
            if self.kwargs.get('incl_light', False):
                record_buffers['LUX'] = []
            activity_types = set()
            timestamp_buffers = {key: [] for key in record_buffers}

            with self.zip_file.open(self.LogRecord.DATA_STREAM_FILE_NAME) as binary_stream:
                while True:
                    record, decoded = ActiGraphLogRecordV2.from_stream(binary_stream)
                    if record is None:
                        break

                    record_type_name = record.record_type
                    if record_type_name in ('ACTIVITY', 'ACTIVITY2') and record.header.payload_size == 1:
                        continue  # skip 1-byte USB docking artifacts

                    if record_type_name in record_buffers:
                        if decoded is None:
                            raise ValueError(f"{record_type_name} record is not decoded.")

                        start_ts = pd.Timestamp(record.header.timestamp, unit='s')
                        n_samples = decoded.shape[0] if record_type_name.startswith('ACTIVITY') else self.sample_rate
                        record_timestamps = self._create_timestamps(start_ts, n_samples, self.sample_rate)
                        if record_type_name in ('ACTIVITY', 'ACTIVITY2'):
                            if decoded.shape[0] != self.sample_rate:
                                print(record)
                                raise ValueError(
                                    f"(io: {self.meta['patient_id']}) ValueError:"
                                    f"Unexpected sample count in {record_type_name} record:"
                                    f"expected {self.sample_rate}, got {decoded.shape[0]}.")
                            activity_types.add(record_type_name)
                        else:  # for battery and lux, only average over 1s is recorded, repeated over full record length
                            decoded = self._pad_average_value_over_record_length(decoded, self.sample_rate)

                        record_buffers[record_type_name].append(decoded)
                        timestamp_buffers[record_type_name].append(record_timestamps)

            if len(activity_types) == 0:
                raise RuntimeError("No activity records found.")
            elif len(activity_types) == 1:
                activity_type = activity_types.pop()
            else:
                raise RuntimeError(f"Mixed activity types found: {activity_types} — expected only one.")

            activity_data = self._calibrate_acceleration(  # apply manufacture calibration if available, noqa
                np.vstack(record_buffers.get(activity_type)), record_type_name)
            activity_timestamps = pd.DatetimeIndex(np.concatenate(timestamp_buffers[activity_type]))

            df = pd.DataFrame(activity_data, columns=['x', 'y', 'z'], index=activity_timestamps)
            df.index.name = 'time'

            if ('BATTERY' in record_buffers and record_buffers['BATTERY']) or \
                    ('LUX' in record_buffers and record_buffers['LUX']):
                aux_dfs = {}

                if 'BATTERY' in record_buffers and record_buffers['BATTERY']:
                    battery_data = np.concatenate(record_buffers['BATTERY'])
                    battery_timestamps = pd.DatetimeIndex(np.concatenate(timestamp_buffers['BATTERY']))
                    aux_dfs['battery'] = pd.DataFrame({'battery': battery_data}, index=battery_timestamps)

                if 'LUX' in record_buffers and record_buffers['LUX']:
                    lux_data = np.concatenate(record_buffers['LUX'])
                    lux_timestamps = pd.DatetimeIndex(np.concatenate(timestamp_buffers['LUX']))
                    aux_dfs['lux'] = pd.DataFrame({'lux': lux_data}, index=lux_timestamps)

                raise NotImplementedError("Battery and/or Lux data are available but not yet aligned with"
                                          " timestamps in the activity DataFrame.")

            self.status_ok = 1
            return df, self.device_header

        except Exception as e:
            logger.error(f"(io: {self.meta['patient_id']}) Exception occurred: {e}")
            self.status_ok = 0
            return None

    def _calibrate_acceleration(self, acceleration: np.ndarray, record_type_name: str) -> np.ndarray:
        """Calibrate acceleration data if manufacture calibration data is available. Acceleration has shape (N, XYZ)."""

        isotropic_scale_factor = self._get_acceleration_scale_factor(record_type_name)

        def __try_calibration(label: str, func, *args) -> np.ndarray:
            try:
                result = func(*args)
                self.device_header['calibration_mode'] = label
                return result
            except KeyError as e:
                logger.warning(
                    f"(io: {self.meta['patient_id']}) Missing {label} coefficient: {e}. Using linear scaling.")
                self.device_header['calibration_mode'] = f"{label} failed"
                return acceleration / isotropic_scale_factor

        if self.LogRecord.CALIBRATION_FILE_NAME in self.zip_file.namelist():
            with self.zip_file.open(self.LogRecord.CALIBRATION_FILE_NAME) as fp:
                calibration_info = json.load(fp)

            if calibration_info.get('isCalibrated', True):
                # data is already calibrated, so just return the scaled values in g
                self.device_header['calibration_mode'] = 'pre-calibrated'
                return acceleration / isotropic_scale_factor

            elif calibration_info["calibrationMethod"] == 1:
                return __try_calibration(
                    '6-point static', self._six_point_calibration, acceleration, calibration_info)

            elif calibration_info["calibrationMethod"] == 2:
                return __try_calibration(
                    'sensitivity matrix', self._sensitivity_matrix_calibration,
                    acceleration, calibration_info, self.sample_rate)

            raise NotImplementedError(
                f"(io: {self.meta['patient_id']}) Unknown or unsupported calibration method:"
                f"{calibration_info.get('calibrationMethod')}")

        else:
            logger.warning(f"(io: {self.meta['patient_id']}) {self.LogRecord.CALIBRATION_FILE_NAME}"
                           f"missing, skipping manufacturer calibration. ")
            return acceleration / isotropic_scale_factor

    @staticmethod
    def _sensitivity_matrix_calibration(acceleration: np.ndarray, calibration_info: dict, sample_rate: float):
        """https://github.com/actigraph/pygt3x/blob/c40eaa9f029623ad818f27a77e2e7e7ab50e7b9d/pygt3x/calibration.py"""
        # get offsets and sensitivity coefficients for each axis
        sr = int(sample_rate)
        offset = np.array([
            calibration_info[f"offsetX_{sr}"], calibration_info[f"offsetY_{sr}"], calibration_info[f"offsetZ_{sr}"]
        ], dtype=np.float64).reshape(1, 3)
        keys = [f"sensitivity{suffix}_{sr}" for suffix in ["XX", "YY", "ZZ", "XY", "XZ", "YZ"]]
        s_xx, s_yy, s_zz, s_xy, s_xz, s_yz = [calibration_info[k] for k in keys]

        # construct matrix using ActiGraph transformation rules
        def inv(val): return (val * 0.01) ** -1.0

        def cross(val): return ((val * 0.01 + 250) ** -1.0) - 0.004

        s_matrix = np.array([
            [inv(s_xx), cross(s_xy), cross(s_xz)],
            [cross(s_xy), inv(s_yy), cross(s_yz)],
            [cross(s_xz), cross(s_yz), inv(s_zz)],
        ])
        return (s_matrix @ (acceleration - offset).T).T

    @staticmethod
    def _six_point_calibration(acceleration: np.ndarray, calibration_info: dict):
        # ActiGraph WP 16CPAN03: https://s3.amazonaws.com/actigraphcorp.com/wp-content/uploads/
        # 2017/11/26205816/Scaling_and_Correcting_Accelerometer_Data_Affected_by_16CPAN03.pdf
        pos = [calibration_info[f"positiveZeroGOffset{ax}"] for ax in "XYZ"]
        neg = [calibration_info[f"negativeZeroGOffset{ax}"] for ax in "XYZ"]
        zero = [calibration_info[f"zeroGOffset{ax}"] for ax in "XYZ"]
        # ActiGraph WP 16CPAN03: Eq. 1-3
        gain = np.array([2.0 / (p - n) for p, n in zip(pos, neg)], dtype=np.float64)
        # ActiGraph WP 16CPAN03: Eq. 4-6
        offset = np.array([z * g for z, g in zip(zero, gain)], dtype=np.float64)
        return (acceleration - offset) * gain

    def _load_g3tx_info(self):
        """Parse info.txt inside a GT3X zip archive into a dictionary, with numeric conversion and stripping."""

        def __ticks_to_datetime(ticks: int) -> pd.Timestamp:
            """Convert .NET ticks (100 ns intervals since 0001-01-01) to pandas Timestamp (UTC)."""
            base = datetime(1, 1, 1)
            dt = base + timedelta(microseconds=ticks / 10)
            return pd.Timestamp(dt, tz='UTC')

        _type_map = {
            'Serial Number': str, 'Firmware': str, 'Device Type': str, 'Subject Name': str,
            'Limb': str, 'TimeZone': str,
            'Battery Voltage': lambda v: float(v.replace(",", ".")),
            'Acceleration Max': lambda v: float(v.replace(",", ".")),
            'Acceleration Min': lambda v: float(v.replace(",", ".")),
            'Acceleration Scale': lambda v: float(v.replace(",", ".")),
            'Sample Rate': lambda v: float(v.replace(",", ".")),
            'Board Revision': int, 'Unexpected Resets': int,
            # datetime conversions
            'Start Date': lambda v: __ticks_to_datetime(int(v)), 'Stop Date': lambda v: __ticks_to_datetime(int(v)),
            'Last Sample Time': lambda v: __ticks_to_datetime(int(v)),
            'Download Date': lambda v: __ticks_to_datetime(int(v)),
        }

        info_dict = {}
        with self.zip_file.open('info.txt') as f:
            for line in f:
                decoded = line.decode('utf-8-sig').strip()
                if ':' in decoded:
                    key, value = decoded.split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    cast = _type_map.get(key, str)
                    try:
                        info_dict[key] = cast(value)
                    except Exception:
                        logger.warning(
                            f"(io: {self.meta['patient_id']}) Failed to parse '{key}': '{value}' "
                            f"— storing as string fallback.")
                        info_dict[key] = value

        # Rename key to match usage elsewhere in your code
        if 'Sample Rate' in info_dict:
            info_dict['sample_rate'] = info_dict.pop('Sample Rate')

        return info_dict, info_dict['sample_rate']

    def _get_acceleration_scale_factor(self, activity_type: str) -> float:
        """Determine the scale factor to convert acceleration data to g."""
        if activity_type.upper() == 'ACTIVITY':
            _bit_range = 2 ** 12
        elif activity_type.upper() == 'ACTIVITY2':
            _bit_range = 2 ** 16
        else:
            raise ValueError(f"Unknown activity type '{activity_type}'. Must be in ('ACTIVITY', 'ACTIVITY2').")
        try:  # Option 1: ‘Acceleration_Scale’ in 'info.txt'/header
            scale = self.device_header.get("Acceleration Scale")
            if scale is None:  # if not available, use range to determine scale factor
                rng_min = float(self.device_header.get("Acceleration Min"))
                rng_max = float(self.device_header.get("Acceleration Max"))
                scale = _bit_range / (rng_max - rng_min)  # 12‑bit ADC → 4096 counts
            scale = float(scale)

        except (TypeError, ValueError):  # no scale information in header, fallback to nominal scale
            logger.warning(f"(io: {self.meta['patient_id']}) could not determine scale factor from 'info.txt'. "
                           f"Using nominal scale factor of 256 LBS / g.")
            scale = 256

        return float(scale)

    @property
    def is_in_nhanes_format(self) -> bool:
        """Check if the GT3X file is in the older v1(NHANES) format, i.e., does not contain the 'log.bin' file."""
        return 'log.bin' not in self.zip_file.namelist()

    def _assert_ism_was_disabled(self):
        status = self.ism_was_enabled
        if status is True:
            raise RuntimeError(
                f"Idle‑Sleep‑Mode (ISM) is active in '{Path(self.processing_info['loading']['filepath']).name}'. "
                "This pipeline expects uninterrupted raw acceleration; ISM inserts flat‑line samples, especially at "
                "night, which distorts sleep‑period sensitivity. See JMPB 2024 (PMCID: PMC11524531) for details.")

        if status is None:
            logger.warning(
                f"(io: {self.meta['patient_id']}) Cannot determine ISM status "
                f"for '{self.processing_info['loading']['filepath'].name}'. Proceeding anyway.")

    @property
    def ism_was_enabled(self) -> Optional[bool]:
        """Determine whether Idle Sleep Mode (ISM) was enabled from PARAMETER records."""
        if self.LogRecord.DATA_STREAM_FILE_NAME not in self.zip_file.namelist():
            logger.info(
                f"(io: {self.meta['patient_id']}) No 'log.bin' found — ISM status not available (likely V1 format).")
            return None

        try:
            with self.zip_file.open("log.bin") as stream:
                while True:
                    record, decoded = ActiGraphLogRecordV2.from_stream(stream)
                    if record is None:
                        break  # EOF

                    # scan PARAMETER records for sign of ISM activation (only valid for v2 format)
                    if record.record_type != "PARAMETERS":
                        continue

                    return decoded['ISM_enabled']

        except Exception as e:
            logger.error(f"(io: {self.meta['patient_id']}) ISM detection failed: {e}")
            return None

        return False  # No indication of ISM found

    @staticmethod
    def _create_timestamps(payload_start: pd.Timestamp, n_samples: int, sample_rate: int) -> pd.DatetimeIndex:
        """Generate timestamps for a 1-second record."""
        delta = pd.to_timedelta(np.arange(n_samples) / sample_rate, unit='s')
        return payload_start + delta

    @staticmethod
    def _pad_average_value_over_record_length(value: float, n_samples: int) -> np.ndarray:
        """Repeat a scalar value to match number of samples."""
        return np.full(int(n_samples), value, dtype=np.float32)
