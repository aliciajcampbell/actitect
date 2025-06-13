import calendar
import logging
import mmap
from collections import namedtuple
import struct
import urllib.parse
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Union
import numpy as np
import pandas as pd

from aktiRBD.actimeter.basedevice import BaseDevice
from aktiRBD import utils

__all__ = ['AxivityAx6']

logger = logging.getLogger(__name__)


@utils.optional_njit(cache=True, fastmath=True, nogil=True)
def checksum16(view_u2: np.ndarray) -> int:
    """SIMD* in Numba, Python fallback otherwise."""
    return int(view_u2.sum(dtype=np.uint32) & 0xFFFF)


@dataclass
class Ax6BinaryFormat:
    BLOCK_SIZE: int = 512

    _Header = namedtuple("_Header", "device_frac session_id sequence_id ts_packed light_raw temp_raw "
                                    "events battery_raw rate_code num_axesbps ts_offset sample_cnt", )

    CWA_DTYPES: np.dtype = field(init=False, repr=False, default=np.dtype([
        ('packet_header', '<H'),
        ('packet_length', '<H'),
        ('device_fractional', '<H'),
        ('session_id', '<I'),
        ('sequence_id', '<I'),
        ('timestamp_packed', '<I'),
        ('scale_light', '<H'),
        ('temperature', '<H'),
        ('events', 'B'),
        ('battery', 'B'),
        ('rate_code', 'B'),
        ('num_axes_bps', 'B'),
        ('timestamp_offset', '<h'),
        ('sample_count', '<H'),
        ('raw_data_buffer', np.dtype('V480')),
        ('checksum', '<H')
    ]))

    @staticmethod
    def decode_timestamp(packed: int, return_datetime: bool = False) -> Optional[Union[datetime, int]]:
        """Decode a packed CWA timestamp to datetime or int (Unix time). If `return_datetime=True`,
         returns a datetime.datetime object. Otherwise returns seconds since epoch as int."""
        y = ((packed >> 26) & 0x3F) + 2000
        m = (packed >> 22) & 0x0F
        d = (packed >> 17) & 0x1F
        H = (packed >> 12) & 0x1F
        M = (packed >> 6) & 0x3F
        S = packed & 0x3F
        try:
            dt = datetime(y, m, d, H, M, S)
            return dt if return_datetime else calendar.timegm(dt.timetuple())
        except ValueError:
            return None

    @staticmethod
    def timestamp_to_string(value: Optional[Union[int, float, datetime]]) -> str:
        if isinstance(value, (int, float)):
            value = datetime.utcfromtimestamp(value)
        return value.strftime('%Y-%m-%d %H:%M:%S.000') if value else 'Invalid'

    @staticmethod
    def cwa_parse_metadata(meta_bytes: bytes) -> dict:
        """Parse URL-encoded metadata from CWA block."""
        shorthand = dict(
            _c='Study Centre', _s='Study Code', _i='Investigator', _x='Exercise Code', _v='Volunteer Num',
            _p='Body Location', _so='Setup Operator', _n='Notes', _b='Start time', _e='End time',
            _ro='Recovery Operator', _r='Retrieval Time', _co='Comments', _sc='Subject Code', _se='Sex',
            _h='Height', _w='Weight', _ha='Handedness', _sn='Subject Notes')
        raw = meta_bytes.rstrip(b"\x00\x20\xff").decode("ascii", errors="ignore")
        pairs = raw.split("&")
        metadata = {}

        for pair in pairs:
            if not pair:
                continue
            parts = pair.split("=", 1)
            key = parts[0].replace("+", " ")
            val = parts[1].replace("+", " ") if len(parts) > 1 else ""
            key = shorthand.get(key, key)
            metadata[key] = urllib.parse.unquote(val)

        return metadata

    @staticmethod
    def _checksum(data: bytes) -> int:
        # view the 512-byte block as uint16s – no copy
        view = np.frombuffer(data, dtype='<u2', count=256)
        return checksum16(view)

    @staticmethod
    @utils.optional_njit(cache=True, fastmath=True, nogil=True)
    def _dword_unpack(value: int):
        """Unpack one DWORD-packed triaxial value (scalar path)."""
        exp = value >> 30
        x = (((value & 0x3FF) ^ 0x200) - 0x200) << exp
        y = ((((value >> 10) & 0x3FF) ^ 0x200) - 0x200) << exp
        z = ((((value >> 20) & 0x3FF) ^ 0x200) - 0x200) << exp
        return x, y, z

    def _unpack_fixed_header(self, block: bytes) -> _Header:
        """Return the 11 fixed header fields as a namedtuple."""
        return self._Header(*struct.unpack_from("<H I I I H H B B B B h H", block, 4))

    def _extract_samples(self, block: bytes, hdr: _Header, accel_ax: int, gyro_ax: int, mag_ax: int, accel_unit: float,
                         gyro_unit: float, mag_unit: float, bytes_axis: int, channels: int, ) -> dict:
        """Return dict with samplesAccel / Gyro / Mag if requested."""
        if hdr.sample_cnt == 0:
            return {}

        off, out = 30, {}  # sample buffer start
        if bytes_axis == 0 and channels == 3:  # DWORD-packed accel
            accel = []
            for i in range(hdr.sample_cnt):
                v = struct.unpack_from("<I", block, off + 4 * i)[0]
                accel.append([x / accel_unit for x in self._dword_unpack(v)])
            out["samplesAccel"] = accel

        elif bytes_axis == 2:
            def __triplet(base, unit):
                return (
                    (block[base] | (block[base + 1] << 8)) / unit,
                    (block[base + 2] | (block[base + 3] << 8)) / unit,
                    (block[base + 4] | (block[base + 5] << 8)) / unit,)

            step = 2 * channels
            if accel_ax >= 0:
                out['samplesAccel'] = [__triplet(off + i * step + 2 * accel_ax, accel_unit)
                                       for i in range(hdr.sample_cnt)]
            if gyro_ax >= 0:
                out['samplesGyro'] = [__triplet(off + i * step + 2 * gyro_ax, gyro_unit) for i in range(hdr.sample_cnt)]
            if mag_ax >= 0:
                out['samplesMag'] = [__triplet(off + i * step + 2 * mag_ax, mag_unit) for i in range(hdr.sample_cnt)]

        return out

    def parse_cwa_data(self, block: bytes, extractData: bool = False) -> dict:
        """Parse one 512-byte AX3/AX6 data block."""
        if (len(block) < self.BLOCK_SIZE or block[:2] != b"AX"
                or struct.unpack_from("<H", block, 2)[0] != 508 or self._checksum(block[:512])):
            return {}

        hdr = self._unpack_fixed_header(block)
        # ── derived constants ─────────────────────────────────────────────
        freq = hdr.ts_offset if hdr.rate_code == 0 else 3200 / (1 << (15 - (hdr.rate_code & 0x0F)))
        timestamp = self.decode_timestamp(hdr.ts_packed)
        ts_offset = hdr.ts_offset
        if hdr.device_frac & 0x8000:
            frac = (hdr.device_frac & 0x7FFF) << 1
            ts_offset += (frac * int(freq)) >> 16
            timestamp += frac / 65536.0

        channels = (hdr.num_axesbps >> 4) & 0x0F
        bytes_axis = hdr.num_axesbps & 0x0F
        bytes_sample = 4 if (bytes_axis == 0 and channels == 3) else bytes_axis * channels
        samples_blk = 480 // bytes_sample

        accel_unit = 1 << (8 + ((hdr.light_raw >> 13) & 0x07))
        gyro_rng = 8000 // (1 << ((hdr.light_raw >> 10) & 0x07)) if (hdr.light_raw >> 10) & 0x07 else 2000
        gyro_unit = 32768.0 / gyro_rng
        accel_rng = 16 if hdr.rate_code == 0 else 16 >> (hdr.rate_code >> 6)
        mag_unit = 16

        accel_ax = gyro_ax = mag_ax = -1
        if channels >= 6:
            gyro_ax, accel_ax = 0, 3
            mag_ax = 6 if channels >= 9 else -1
        elif channels >= 3:
            accel_ax = 0

        data = dict(deviceFractional=hdr.device_frac, sessionId=hdr.session_id, sequenceId=hdr.sequence_id,
                    timestamp=timestamp, timestampOffset=ts_offset, timestampTime=self.timestamp_to_string(timestamp),
                    light=hdr.light_raw & 0x3FF, temperature=hdr.temp_raw * 75.0 / 256 - 50, events=hdr.events,
                    battery=(hdr.battery_raw + 512.0) * 6000 / 1024 / 1000.0, rateCode=hdr.rate_code,
                    numAxesBPS=hdr.num_axesbps, sampleCount=hdr.sample_cnt, frequency=freq, channels=channels,
                    bytesPerAxis=bytes_axis, bytesPerSample=bytes_sample, samplesPerBlock=samples_blk,
                    estimatedFirstSampleTime=timestamp - (ts_offset / freq),
                    estimatedAfterLastSampleTime=timestamp - (ts_offset / freq) + samples_blk / freq)

        if accel_ax >= 0:
            data.update(accelAxis=accel_ax, accelRange=accel_rng, accelUnit=accel_unit)
        if gyro_ax >= 0:
            data.update(gyroAxis=gyro_ax, gyroRange=gyro_rng, gyroUnit=gyro_unit)
        if mag_ax >= 0:
            data.update(magAxis=mag_ax, magRange=32768 / mag_unit, magUnit=mag_unit)

        if extractData:
            data.update(self._extract_samples(
                block, hdr, accel_ax, gyro_ax, mag_ax, accel_unit, gyro_unit, mag_unit, bytes_axis, channels, ))

        return data


class AxivityAx6(BaseDevice):
    """Adapted from [1]. https://github.com/openmovementproject/openmovement-python/ at
    25f546389af518ff184818ca96a4f92e9275b951"""

    def __init__(self, path_to_cwa: Path, patient_id: str, kwargs: dict = None,
                 AX6FORMAT: Ax6BinaryFormat = Ax6BinaryFormat(), legacy_mode: bool = False):
        super().__init__(filepath=path_to_cwa, patient_id=patient_id)
        self.AX6FORMAT = AX6FORMAT
        self.legacy_mode = legacy_mode  # flag to reproduce legacy code in [1] by matching timestamp interpolation error
        # (duplicate at end and beginning) and ignoring invalid checksums
        self._ts_lut = self._build_fast_ts_lut()
        self.kwargs = kwargs if kwargs \
            else {'include_gyro': False, 'include_temperature': False, 'include_light': False, 'include_mag': False}

    def __str__(self):
        return f"AxivityAx6(patient_ID='{self.meta['patient_id']}')"

    def _parse_binary_to_df(self, ax6_logging_threshold_d: int = 8, header_only: bool = False):
        """ Parse the binary file and return it as pd.Dataframe.
        Parameters:
            :param ax6_logging_threshold_d: (int, Optional) If not None, use the 'LoggingStartTime' value from
                the binary header to trim the data. Useful if device was switched on to early and data has several
                days of non-wear before data taking. Only is applied if the data spans longer than
                'ax6_logging_threshold_d' days. The default is set to 8 but might be adjusted for longer data-taking.
            :param header_only: (bool, Optional) If True, only the binary header is returned. Default is False.
        Returns:
            :return: (pd.DataFrame) the raw data parsed as DataFrame.
        """
        logger.info(f"(io: {self.meta['patient_id']}) loading from '{self.processing_info['loading']['filepath']}'.")

        with self.processing_info['loading']['filepath'].open('rb') as f:
            buffer = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

            # parse the header, i.e., the first block
            header_block = buffer[:self.AX6FORMAT.BLOCK_SIZE]
            header = self._parse_header(header_block)
            if header_only:
                return pd.DataFrame(), header

            # validate data blocks and get data block offset
            packet_len = header.get('packetLength')
            if not packet_len:
                raise ValueError("Missing 'packetLength' in header, cannot determine data offset.")
            data_offset = \
                ((packet_len + self.AX6FORMAT.BLOCK_SIZE - 1) // self.AX6FORMAT.BLOCK_SIZE
                 ) * self.AX6FORMAT.BLOCK_SIZE

            # Read first data block (after header) to determine data format
            first_data_block = buffer[data_offset:data_offset + self.AX6FORMAT.BLOCK_SIZE]
            if len(first_data_block) < self.AX6FORMAT.BLOCK_SIZE or first_data_block[:2] != b"AX":
                raise ValueError("First data block is invalid or missing.")

            data_format = self.AX6FORMAT.parse_cwa_data(first_data_block)
            if ('channels' not in data_format or data_format['channels'] < 1
                    or 'samplesPerBlock' not in data_format
                    or data_format['samplesPerBlock'] <= 0):
                raise Exception('Unexpected data format')

            block_array, valid_block_mask, block_info = \
                self._parse_binary_data_blocks(buffer[data_offset:], data_format)
            samples, labels = self._decode_binary_blocks(block_array, valid_block_mask, data_format)

        # create the DataFrame
        df = pd.DataFrame(samples[:, 1:], columns=labels[1:])
        time_ns = (samples[:, 0] * 1_000_000_000).astype('int64')
        df.insert(0, labels[0], time_ns.astype('datetime64[ns]'), allow_duplicates=True)
        df.set_index('time', inplace=True)
        if ax6_logging_threshold_d and header.get('loggingStartTime') and not header_only:
            logging_start_time = pd.to_datetime(header.get('loggingStartTime'), errors='coerce')
            if logging_start_time is not pd.NaT:
                data_span_days = (df.index[-1] - df.index[0]).total_seconds() / (24 * 60 * 60)
                if data_span_days > ax6_logging_threshold_d:  # trim it using 'loggingStartTime'
                    df = df[df.index >= logging_start_time]
                    logger.info(f"(io: {self.meta['patient_id']}) using 'loggingStartTime'"
                                f"({pd.Timestamp(header['loggingStartTime'])}) to trim raw data.")
        else:
            if ax6_logging_threshold_d and not header_only:
                logger.warning(f"(io: {self.meta['patient_id']}) 'loggingStartTime' not found in header;"
                               f"cannot apply logging start time cutoff.")

        return df, header

    @staticmethod
    def _iter_data_blocks(buffer, offset: int, block_size: int):
        for i in range(offset, len(buffer), block_size):
            block = buffer[i:i + block_size]
            if len(block) == block_size:
                yield block

    def _parse_header(self, block: bytes) -> dict:
        """Parse the 512-byte CWA header block."""
        if len(block) != self.AX6FORMAT.BLOCK_SIZE or block[:2] != b"MD":
            raise ValueError("Invalid CWA header block")

        u8 = lambda o: struct.unpack('B', block[o:o + 1])[0]
        u16 = lambda o: struct.unpack('<H', block[o:o + 2])[0]
        u32 = lambda o: struct.unpack('<I', block[o:o + 4])[0]

        header = {'packetLength': u16(2), 'deviceId': u16(5), 'sessionId': u32(7)}
        upper_id = u16(11)
        if upper_id != 0xFFFF:
            header['deviceId'] |= upper_id << 16

        # Timestamps
        header['loggingStart'] = self.AX6FORMAT.decode_timestamp(u32(13)) if u32(13) else 0
        header['loggingEnd'] = self.AX6FORMAT.decode_timestamp(u32(17)) if u32(17) else 0
        header['lastChange'] = self.AX6FORMAT.decode_timestamp(u32(37)) if u32(37) else 0
        header['loggingStartTime'] = self.AX6FORMAT.timestamp_to_string(header['loggingStart'])
        header['loggingEndTime'] = self.AX6FORMAT.timestamp_to_string(header['loggingEnd'])
        header['lastChangeTime'] = self.AX6FORMAT.timestamp_to_string(header['lastChange'])

        # Device & firmware
        hw_type = u8(4)
        header.update({
            'hardwareType': hw_type,
            'deviceType': 'AX3' if hw_type in (0x00, 0xFF, 0x17) else 'AX6' if hw_type == 0x64 else f"0x{hw_type:02x}",
            'firmwareRevision': u8(41),
            'flashLed': 0 if u8(35) == 0xFF else u8(35),
        })

        # Sensor config
        sensor_cfg = u8(35)
        header['gyroRange'] = 0 if sensor_cfg in (0x00, 0xFF) else 8000 / 2 ** (sensor_cfg & 0x0F)

        # Sampling
        rate_code = u8(36)
        header['sampleRate'] = 3200 / (1 << (15 - (rate_code & 0x0F)))
        header['accelRange'] = 16 >> (rate_code >> 6)

        # Misc
        header['loggingCapacity'] = u32(21)
        header['metadata'] = {k: v for k, v in self.AX6FORMAT.cwa_parse_metadata(block[64:512]).items() if k}

        header['sample_rate'] = header.pop('sampleRate')
        return header

    def _parse_binary_data_blocks(self, data_buffer: bytes, data_format: dict) -> tuple[np.ndarray, np.ndarray, dict]:
        """Parse all data blocks and return block array and validity mask using NumPy only."""
        block_array = np.frombuffer(data_buffer, dtype=self.AX6FORMAT.CWA_DTYPES)
        n_blocks = len(block_array)

        # Checksum calculation
        block_size_words = self.AX6FORMAT.BLOCK_SIZE // 2
        np_words = np.frombuffer(data_buffer, dtype=np.dtype('<H'))
        np_block_words = np.reshape(np_words, (-1, block_size_words))
        checksum_sum = np.add.reduce(np_block_words, axis=1, dtype=np.uint16)

        # Validity check
        valid_block = self._filter_valid_blocks(block_array, checksum_sum, data_format, legacy_mode=self.legacy_mode)

        info = {'n_blocks': n_blocks, 'valid_blocks': np.count_nonzero(valid_block)}

        return block_array, valid_block, info

    @staticmethod
    def _build_fast_ts_lut() -> np.ndarray:
        SECONDS_PER_DAY = 86400
        DAYS_IN_MONTH = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31, 0, 0, 0]
        lut = np.zeros(1024, dtype=np.uint32)
        secs = 946684800  # 2000-01-01
        for yr in range(64):
            for mo in range(16):
                idx = (yr << 4) + mo
                lut[idx] = secs - SECONDS_PER_DAY  # minus one day
                days = DAYS_IN_MONTH[mo] + (yr % 4 == 0 and mo == 2)
                secs += days * SECONDS_PER_DAY
        return lut

    @staticmethod
    def _fast_timestamp_vectorized(packed: np.ndarray, fast_ts_lut) -> np.ndarray:
        ym = (packed >> 22) & 0x3FF
        day = (packed >> 17) & 0x1F
        hour = (packed >> 12) & 0x1F
        minu = (packed >> 6) & 0x3F
        sec = packed & 0x3F
        return (fast_ts_lut[ym]
                + ((day * 24 + hour) * 60 + minu) * 60 + sec).astype(np.float64)

    @staticmethod
    def _filter_valid_blocks(block_array, checksum_sum, data_format, *, legacy_mode=False):
        """ Build a Boolean mask of valid blocks and warn if any blocks are excluded.
        Parameters
            :param block_array: (np.ndarray) Structured array with the per-block fields.
            :param checksum_sum: (np.ndarray) 1-D array with sector checksums.
            :param data_format: (dict) Dict holding 'numAxesBPS' and 'rateCode'.
            :param legacy_mode: (bool, optional) If True, the checksum test is skipped.
        Returns
            :return: (np.ndarray) Boolean mask (True = valid)."""
        # Boolean tests for every criterion
        crit = {
            "header": block_array["packet_header"] == 0x5841,
            "length": block_array["packet_length"] == 508,
            "axes/bps": block_array["num_axes_bps"] == data_format["numAxesBPS"],
            "rate": block_array["rate_code"] == data_format["rateCode"],
            "checksum": (checksum_sum == 0) | legacy_mode,
        }

        # Overall validity mask
        valid = np.logical_and.reduce(list(crit.values()))
        assert valid.shape[0] == block_array.shape[0], "Mismatch in validity mask length"

        # Emit warnings for exclusions
        for name, ok in crit.items():
            if name == "checksum" and legacy_mode:  # checksum ignored
                continue
            n_excl = np.count_nonzero(~ok)
            if n_excl:
                logger.warning("Excluded %d blocks – failed %s check", n_excl, name)

        return valid

    def _decode_binary_blocks(self, block_array: np.ndarray, valid_block_mask: np.ndarray, data_format: dict
                              ) -> tuple[np.ndarray, list]:

        """Convert validated CWA blocks into a per-sample NumPy array plus labels."""
        if not np.any(valid_block_mask):
            raise ValueError("No valid data blocks found.")

        blocks = block_array[valid_block_mask]
        samp_per_block = data_format['sampleCount']
        num_blocks = len(blocks)
        total_samples = samp_per_block * num_blocks

        channels = data_format['channels']
        bytes_per_axis = data_format['bytesPerAxis']

        accel_unit = data_format.get('accelUnit', 1)
        accel_axis = data_format.get('accelAxis', 0)
        gyro_unit = data_format.get('gyroUnit', 1)
        gyro_axis = data_format.get('gyroAxis', -1)

        include_gyro = self.kwargs.get('include_gyro', False)
        include_light = self.kwargs.get('include_light', False)
        include_temperature = self.kwargs.get('include_temperature', False)

        n_cols = 4  # time + accel
        if include_gyro:
            n_cols += 3
        if include_light:
            n_cols += 1
        if include_temperature:
            n_cols += 1

        sample_values = np.zeros((total_samples, n_cols), dtype=np.float64)
        labels = ['time', 'x', 'y', 'z']
        if include_gyro:
            labels += ['gyro_x', 'gyro_y', 'gyro_z']
        if include_light:
            labels += ['light']
        if include_temperature:
            labels += ['temperature']

        ts_sec = self._fast_timestamp_vectorized(blocks['timestamp_packed'], self._ts_lut)
        raw_offsets = blocks["timestamp_offset"].astype(np.int32).copy()
        frac16 = blocks["device_fractional"]
        global_frac_flag = bool(data_format["deviceFractional"] & 0x8000)
        offsets = raw_offsets.copy()
        if global_frac_flag:
            tfrac = (frac16.astype(np.int32) & 0x7FFF) * 2
            ts_sec += tfrac / 65536.0

        if self.legacy_mode:
            logger.info(f"(io: {self.meta['patient_id']}) Using legacy mode timestamps ('legacy_mode=True').")
            timestamp_index = (np.arange(num_blocks) * samp_per_block) + offsets
            full_idx = np.arange(total_samples, dtype=np.int64)
            full_ts = np.interp(full_idx, timestamp_index, ts_sec)
            block_idx = timestamp_index
        else:
            block_idx = (np.arange(num_blocks) * samp_per_block) + offsets
            if num_blocks > 1 and block_idx[0] > 0:  # correct extrapolation issue for first and last timestamps
                delta_idx = block_idx[1] - block_idx[0]
                if delta_idx:
                    delta_t = ts_sec[1] - ts_sec[0]
                    ts_sec[0] -= delta_t / delta_idx * block_idx[0]
                    block_idx[0] = 0
            full_idx = np.arange(total_samples, dtype=np.int64)
            full_ts = np.interp(full_idx, block_idx, ts_sec)

        sample_values[:, 0] = full_ts

        raw_bytes = blocks['raw_data_buffer'].flatten().tobytes()
        if bytes_per_axis == 0 and channels == 3:
            raw_dw = np.frombuffer(raw_bytes, dtype='<I', count=total_samples)
            exp = raw_dw >> 30
            sample_values[:, 1] = ((((raw_dw >> 0) & 0x3ff) ^ 0x200) - 0x200) << exp
            sample_values[:, 2] = ((((raw_dw >> 10) & 0x3ff) ^ 0x200) - 0x200) << exp
            sample_values[:, 3] = ((((raw_dw >> 20) & 0x3ff) ^ 0x200) - 0x200) << exp
        elif bytes_per_axis == 2:
            raw_w = np.frombuffer(raw_bytes, dtype='<h').reshape((-1, channels))
            sample_values[:, 1:4] = raw_w[:, accel_axis:accel_axis + 3]
            if include_gyro and gyro_axis >= 0:
                sample_values[:, 4:7] = raw_w[:, gyro_axis:gyro_axis + 3] / gyro_unit
        else:
            raise NotImplementedError("Unsupported packing format")

        sample_values[:, 1:4] /= accel_unit

        if include_light:
            logger.warning('noqa for light sensor parsing.')
            light_raw = (blocks['scale_light'] & 0x3ff).astype(np.float32)
            tgt = -2 if include_temperature else -1
            sample_values[:, tgt] = np.interp(full_idx, block_idx, light_raw)

        if include_temperature:
            logger.warning('noqa for temperature sensor parsing.')
            temp_raw = ((blocks['temperature'] & 0x3ff) * (75.0 / 256) - 50).astype(np.float32)
            sample_values[:, -1] = np.interp(full_idx, block_idx, temp_raw)

        return sample_values, labels
