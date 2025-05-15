import logging
import struct
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Dict, Any, ClassVar
from typing import Optional, BinaryIO

import numpy as np

__all__ = ['LogRecordV1', 'LogRecordV2']

logger = logging.getLogger(__name__)


@dataclass
class LogRecord(ABC):
    """Each LogRecord consists of a header, the payload carrying the data and a checksum. The header determines
    what kind of data (e.g. activity, battery, etc.) is contained in the payload. Subclasses implement the decoders for
    AGDC and NHANES formats."""

    header: 'LogRecord.Header'
    payload: 'LogRecord.Payload'
    checksum: 'LogRecord.Checksum'

    record_type: str

    @dataclass
    class Header:
        separator: int
        type: int
        timestamp: int
        payload_size: int

    @dataclass
    class Payload:
        data: bytes

    @dataclass
    class Checksum:
        checksum: int
        is_valid: bool

    @classmethod
    @abstractmethod
    def from_stream(cls, stream: BinaryIO) -> Optional['LogRecord']:
        pass


@dataclass
class LogRecordV1(LogRecord, ABC):  # NHANES
    @classmethod
    def from_stream(cls, stream: BinaryIO) -> Optional['LogRecord']:
        raise NotImplementedError("LogRecordV1 is not implemented (NHANES format).")


@dataclass
class LogRecordV2(LogRecord):  # AGDC
    """https://github.com/actigraph/GT3X-File-Format"""
    DATA_STREAM_FILE_NAME: str = "log.bin"
    CALIBRATION_FILE_NAME: str = 'calibration.json'
    METADATA_FILE_NAME: str = "info.txt"
    RECORD_SEPARATOR: int = 0x1E
    HEADER_STRUCT_FORMAT: str = "<B B I H"

    RECORD_TYPES: ClassVar[Dict[str, int]] = {
        'ACTIVITY': 0x00,  # 1s of raw activity samples in 12-bit and xyz order
        'BATTERY': 0x02,  # voltage in millivolts as little-endian unsigned short (2 bytes)
        'EVENT': 0x03,  # used for internal debugging, e.g., create record with payload 0x08 = ISM on, 0x09 = ISM off
        'HEART_RATE_BPM': 0x04,  # average beats per minute, one byte unsigned integer
        'LUX': 0x05,  # lux value as little-endian unsigned short (2 bytes)
        'METADATA': 0x06,  # arbitrary metadata, first record in every log contains subject data in JSON format
        'TAG': 0x07,  # 13 Byte Serial, 1 Byte Tx Power, 1 Byte (signed) RSSI
        'EPOCH': 0x09,  # 60-second epoch data
        'HEART_RATE_ANT': 0x0B,  # Heart Rate RR information from ANT+ sensor.
        'EPOCH2': 0x0C,  # 60-second epoch data
        'CAPSENSE': 0x0D,  # Capacitive sense data
        'HEART_RATE_BLE4': 0x0E,  # Bluetooth heart rate information (BPM and RR). This is a Bluetooth standard format.
        'EPOCH3': 0x0F,  # 60-second epoch data
        'EPOCH4': 0x10,  # 60-second epoch data
        'PARAMETERS': 0x15,  # Records various configuration parameters and device attributes on initialization.
        'SENSOR_SCHEMA': 0x18,  # This record allows dynamic definition of a SENSOR_DATA record format.
        'SENSOR_DATA': 0x19,  # This record stores sensor data according to a SENSOR_SCHEMA definition.
        'ACTIVITY2': 0x1A,  # One second of raw activity samples as little-endian signed-shorts in XYZ order.
    }

    @classmethod
    def _register_decoders(cls) -> Dict[int, Callable[[bytes, int], Any]]:
        return {
            0x00: cls._decode_activity,  # 'ACTIVITY' i.e. early AGDC, 12 bit
            0x02: cls._decode_battery,
            # 0x03: cls._decode_event,
            # 0x04: cls._decode_heart_rate_bpm,
            0x05: cls._decode_lux,
            # 0x06: cls._decode_metadata,
            # 0x07: cls._decode_tag,
            # 0x09: cls._decode_epoch,
            # 0x0B: cls._decode_heart_rate_ant,
            # 0x0C: cls._decode_epoch,
            # 0x0D: cls._decode_capsense,
            # 0x0E: cls._decode_heart_rate_ble,
            # 0x0F: cls._decode_epoch,
            # 0x10: cls._decode_epoch,
            0x15: cls._decode_parameters,
            # 0x18: cls._decode_sensor_schema,
            # 0x19: cls._decode_sensor_data,
            0x1A: cls._decode_activity2,  # 'ACTIVITY2' i.e. later AGDC (2014+), 16 bit
        }

    @classmethod
    def from_stream(cls, stream: BinaryIO) -> Optional['LogRecord']:
        try:
            # first 8-bytes are header
            header_bytes = stream.read(8)
            if len(header_bytes) < 8:
                return None, None  # EOF or corrupted
            separator, type_code, timestamp, payload_size = struct.unpack(cls.HEADER_STRUCT_FORMAT, header_bytes)

            # the next payload_size bytes are payload and last byte is checksum
            payload_bytes = stream.read(payload_size)
            checksum_byte = stream.read(1)

            # sanity check
            if len(payload_bytes) < payload_size or len(checksum_byte) < 1:
                raise ValueError("Incomplete payload or missing checksum")

            # create the log records instance and the submodules
            header = cls.Header(separator, type_code, timestamp, payload_size)
            payload = cls.Payload(payload_bytes)
            checksum = cls.Checksum(
                checksum=checksum_byte[0], is_valid=checksum_byte[0] == cls._compute_checksum(header, payload_bytes))
            if not checksum.is_valid:
                raise ValueError('Invalid checksum.')

            record_type = cls._get_name_from_code(type_code)
            record = cls(header=header, payload=payload, checksum=checksum, record_type=record_type)

            # get the correct recorder for the log-records data type
            decoder = cls._register_decoders().get(type_code)
            if decoder:
                decoded = decoder(payload_bytes)
                return record, decoded
            else:  # return None, error handling elsewhere
                return record, None
        except Exception as e:
            raise RuntimeError(f"Failed to parse LogRecordV2: {e}") from e

    @staticmethod
    def _decode_activity(payload: bytes, _: int) -> np.ndarray:
        """https://github.com/actigraph/GT3X-File-Format/blob/main/LogRecords/Activity.md"""

        def __decode_bitpacked_activity_xyz12(_payload: bytes) -> np.ndarray:
            """Decode 12-bit packed activity samples from GT3X binary format.
            Activity data is encoded in a bit-packed format: each group of 3 bytes contains
            two 12-bit signed samples. Every three decoded samples represent a full XYZ vector.
            Values above 2047 are treated as signed 12-bit integers and converted accordingly.
            This implementation handles edge cases (e.g., payloads not divisible by 3) by
            zero-padding the input to avoid alignment issues.
            https://github.com/actigraph/pygt3x/blob/c40eaa9f029623ad818f27a77e2e7e7ab50e7b9d/pygt3x/activity_payload.py
            Parameters:
                :param _payload: (bytes) Raw GT3X activity payload from a record of type 0x00 (ACTIVITY).
            Returns:
                :return: (np.ndarray) An (N, 3) array of signed int16 acceleration samples in XYZ order."""
            data = np.frombuffer(_payload, dtype=np.uint8)
            if data.size % 3 == 1:
                data = np.pad(data, (0, 2), constant_values=0)
            a, b, c = np.reshape(data, (-1, 3)).astype(np.uint16).T
            s1 = (a << 4) | (b >> 4)
            s2 = ((b & 0x0F) << 8) | c
            samples = np.vstack((s1, s2)).T.reshape(-1, 3)
            samples[samples > 2047] += 61440  # convert from unsigned to signed 12-bit
            return samples.astype(np.int16)

        # Unpack 12-bit signed samples in YXZ order
        raw = __decode_bitpacked_activity_xyz12(payload)
        return raw[:, [1, 0, 2]]  # YXZ → XYZ

    @staticmethod
    def _decode_activity2(payload: bytes) -> np.ndarray:
        """https://github.com/actigraph/GT3X-File-Format/blob/main/LogRecords/Activity2.md"""
        if (len(payload) % 6) != 0:
            payload = payload[: -(len(payload) % 6) + 1]
        return np.frombuffer(payload, dtype=np.int16).reshape((-1, 3))

    @staticmethod
    def _decode_battery(payload: bytes) -> float:
        """Decode 2-byte little-endian unsigned short representing battery voltage in mV."""
        return struct.unpack("<H", payload)[0]

    @staticmethod
    def _decode_lux(payload: bytes, _: int) -> float:
        """Decode 2-byte little-endian unsigned short representing Lux. Calibration needed?"""
        return struct.unpack("<H", payload)[0]

    @staticmethod
    def _decode_parameters(payload: bytes) -> dict:
        """Decode configuration parameters from type 0x15 payload.
        Supports both short (2-byte) and long (8-byte) parameter blocks.
        Returns:
            dict: Parsed parameter IDs and values, including 'ISM_enabled' if applicable."""
        out = {}

        if len(payload) == 2:  # case 1: short format (2 bytes) – legacy
            param_id = payload[0]
            value = payload[1]
            out['format'] = 'short_2bytes'
            out[f'param_{param_id:#x}'] = value
            if param_id == 0xD1:  # PARAM_ID_SHORT
                out['ISM_enabled'] = bool(value & 0x01)  # BITMASK_SHORT

        elif len(payload) >= 8:  # case 2: long format (≥8 bytes, multiple of 8) – standard AGDC
            blocks = payload[:len(payload) - (len(payload) % 8)]  # trim excess
            params = np.frombuffer(blocks, dtype="<u8")
            out['format'] = 'long_8bytes'
            for param in params:
                # params are not encoded so far
                # (see https://github.com/actigraph/GT3X-File-Format/blob/main/LogRecords/Parameters.md
                # if needed at some point)
                addr = np.frombuffer(param.tobytes(), dtype="<u1")
                param_id = addr[2]
                value = addr[4]
                out[f"param_{param_id:#x}"] = value
                if param_id == 0x02:  # PARAM_ID_LONG
                    out['ISM_enabled'] = bool(value & 0x04)  # BITMASK_LONG

        return out

    @classmethod
    def _get_decoder(cls, name: str) -> Callable[[bytes, int], Any]:
        """Return decoder by record name (e.g., 'ACTIVITY')."""
        name_upper = name.upper()
        if name_upper not in cls.RECORD_TYPES:
            available = ', '.join(sorted(cls.RECORD_TYPES.keys()))
            raise KeyError(f"Unknown record type name '{name}'. Available types: {available}")

        type_code = cls._get_type_code(name_upper)
        decoder = cls._register_decoders().get(type_code)
        if decoder is None:
            available = ', '.join(f"{cls._get_name_from_code(k)} ({k:#04x})" for k in cls._register_decoders().keys())
            raise KeyError(f"No decoder registered for '{name_upper}' (code {type_code:#04x}). "
                           f"Available decoders: {available}")
        return decoder

    @classmethod
    def _get_type_code(cls, name: str) -> int:
        """Return record type code (e.g., 0x00) from name like 'ACTIVITY'."""
        return cls.RECORD_TYPES[name.upper()]

    @classmethod
    def _get_name_from_code(cls, code: int) -> str:
        """Reverse mapping from byte to name."""
        for name, val in cls.RECORD_TYPES.items():
            if val == code:
                return name
        return f"UNKNOWN_{code:#02x}"

    @staticmethod
    def _compute_checksum(header: LogRecord.Header, payload: bytes) -> int:
        """https://github.com/actigraph/GT3X-File-Format/tree/main"""
        chk = header.separator ^ header.type
        # XOR 4-byte timestamp
        timestamp_bytes = header.timestamp.to_bytes(4, byteorder="little")
        for b in timestamp_bytes:
            chk ^= b
        # XOR 2-byte payload size
        size_bytes = header.payload_size.to_bytes(2, byteorder="little")
        for b in size_bytes:
            chk ^= b
        # XOR payload bytes
        for b in payload:
            chk ^= b
        # Return one's complement
        return ~chk & 0xFF
