import logging
import re
import numpy as np
from pathlib import Path
from joblib import Parallel, delayed
import datetime

import pandas as pd

from ..basedevice import BaseDevice
from ... import utils

__all__ = ['GENEActiv']

logger = logging.getLogger(__name__)


class GENEActiv(BaseDevice):
    """ Subclass of BaseDevice representing GENEActive actimeters. Implements the logic to read binary data from hex-16
    format to decimal. Uses parallel processing to parse multiple pages in parallel for significant speedup.
    Binary decoding according to encoding described in GENEActive manual:
    https://activinsights.com/wp-content/uploads/2022/06/GENEActiv-Instructions-for-Use-v1_31Mar2022.pdf. """

    def __init__(self, path_to_bin: Path, patient_id: str, n_jobs_parser: int = -1, kwargs: dict = None):
        super().__init__(filepath=path_to_bin, patient_id=patient_id)

        self.n_jobs_parser = n_jobs_parser
        self._header_section_patterns = {  # sections in header
            "Device Identity": r"Device Identity(.+?)(?=\n[A-Za-z\s]+\n|$)",
            "Device Capabilities": r"Device Capabilities(.+?)(?=\n[A-Za-z\s]+\n|$)",
            "Configuration Info": r"Configuration Info(.+?)(?=\n[A-Za-z\s]+\n|$)",
            "Trial Info": r"Trial Info(.+?)(?=\n[A-Za-z\s]+\n|$)",
            "Subject Info": r"Subject Info(.+?)(?=\n[A-Za-z\s]+\n|$)",
            "Calibration Data": r"Calibration Data(.+?)(?=\n[A-Za-z\s]+\n|$)",
            "Memory Status": r"Memory Status(.+?)(?=\n[A-Za-z\s]+\n|$)"}
        self.kwargs = kwargs if kwargs else {'calibrate_with_gain_and_offset': True,
                                             'incl_light': False, 'incl_battery': False, 'incl_temperature': False,
                                             'incl_button': False}
        self.skipped_pages, self.failed_pages = [], []
        self.mfr_gain, self.mfr_offset = None, None  # manufacture gain and offsets
        self.light_lux, self.light_volts = None, None  # gain and offset for light data

    def __str__(self):
        return f"GENEActiv(patient_ID='{self.meta['patient_id']}')"

    def _parse_binary_to_df(self, resolve_duplicates=True, header_only: bool = False):
        logger.info(f"(io: {self.meta['patient_id']}) loading from '{self.processing_info['loading']['filepath']}'.")
        try:
            with open(self.processing_info['loading']['filepath'], 'rb') as f:
                file_content = f.read()

            # parse header and get data recording start index
            header, _data_start_index = self._parse_bin_file_header(file_content)
            if self.kwargs.get('calibrate_with_gain_and_offset'):  # make gain/offset available for parser
                _cali_data = header.get('calibration_data')
                self.mfr_gain = [_cali_data.get('x_gain'), _cali_data.get('y_gain'), _cali_data.get('z_gain')]
                self.mfr_offset = [_cali_data.get('x_offset'), _cali_data.get('y_offset'), _cali_data.get('z_offset')]

            if self.kwargs.get('incl_light'):  # make light sensor calibration available for parser
                _cali_data = header.get('calibration_data', {})
                self.light_lux, self.light_volts = _cali_data.get('lux'), _cali_data.get('volts')
            if not header_only:
                pages = self._identify_data_pages(file_content[_data_start_index:])  # identify data recording pages
                assert len(pages) == header['memory_status']['number_of_pages'], \
                    (f"Number of extracted pages ({len(pages)}) does not match "
                     f"expected number from file header ({header['memory_status']['number_of_pages']}).")

                df = self._parse_pages_in_parallel(pages)  # process the pages in parallel
                df.set_index('time', inplace=True)
            else:
                df = pd.DataFrame()
            self.status_ok = 1
            header['sample_rate'] = header['configuration_info'].pop('measurement_frequency')
            return df, header

        except Exception as e:
            logger.error(f"Exception occurred: {e}")
            self.status_ok = 0
            return None

    def _parse_bin_file_header(self, content):
        """ Parse the header to extract metadata and get data recording start index."""
        full_str = content.decode('ascii', errors='ignore')
        # find first data block to only extract header
        data_start_index = full_str.find('Recorded Data')
        if data_start_index == -1:
            raise ValueError("Recorded Data section not found.")
        header_str = full_str[:data_start_index]
        data_start_index = full_str.find('\n', data_start_index) + 1  # adjust index to start after 'Recorded Data' line

        parsed_header = {}
        for _name, _pttrn in self._header_section_patterns.items():  # loop over all header sections
            parsed_header.update({utils.str_to_snake_case(_name): self._extrct_header_section_data(header_str, _pttrn)})

        return parsed_header, data_start_index

    @staticmethod
    def _extrct_header_section_data(full_content: str, section_pattern: str):
        """ for each header section, return values as dict. """

        def _format_value(value):
            """Remove null characters, strip extra whitespace, and converts number strings to float."""
            value = value.replace('\x00', '').strip()
            try:  # attempt if numeric
                return float(value)
            except ValueError:
                return value

        section_content = re.search(section_pattern, full_content, re.DOTALL)
        if section_content:
            section_data = section_content.group(1)
            kv_pairs = re.findall(r"([A-Za-z0-9\s]+):([^\n]*)", section_data)
            return {utils.str_to_snake_case(key): _format_value(value) for key, value in kv_pairs}
        return {}

    @staticmethod
    def _identify_data_pages(data_content: bytes):
        """ Decode the data section and segment the single data pages.
        Parameters:
            :param data_content: (bytes) Content of the file, minus the header.
        Returns:
            :return: (list[tuple]) List of len=n_pages where each entry is tuple of (page_index, page_content). """
        data_str = data_content.decode('ascii', errors='ignore')
        lines = data_str.splitlines(keepends=True)  # keep line endings
        blocks = []
        current_index, block_counter = 0, 0

        while current_index < len(lines):
            line = lines[current_index]
            if 'Sequence Number' in line:
                block_start = current_index
                next_block_index = current_index + 1  # find where the next block starts
                while next_block_index < len(lines) and 'Sequence Number' not in lines[next_block_index]:
                    next_block_index += 1
                block_end = next_block_index
                block_content = ''.join(lines[block_start:block_end])
                blocks.append((block_counter, block_content))
                block_counter += 1
                current_index = next_block_index
            else:
                current_index += 1

        return blocks

    def _parse_pages_in_parallel(self, pages: list[tuple]):
        """ Parse the binary content of all data pages in parallel.
        Parameters:
            :param pages: (list[tuple]) The list of pages to parse, each as tuple of index and binary content.
        Returns:
            :return: pd.DataFrame containing the parsed data of all pages."""
        num_workers = utils.get_num_workers(self.n_jobs_parser)
        logger.info(f"Processing {len(pages)} pages with {num_workers} workers...")

        parsed_pages = Parallel(n_jobs=num_workers)(delayed(self._parse_single_page)(page) for page in pages)
        data_frames = [df for df in parsed_pages if df is not None and not df.empty]
        if data_frames:  # combine results
            combined_df = pd.concat(data_frames, ignore_index=True)
            combined_df.sort_values(by=['page_index', 'sample_index'], inplace=True)
            combined_df.reset_index(drop=True, inplace=True)
            combined_df = combined_df.drop(['page_index', 'sample_index'], axis=1)
            return combined_df  # sorted by correct page index
        else:
            logger.warning(
                f"(io: {self.meta['patient_id']}) could not parse binary pages correctly, returning empty DataFrame!")
            return pd.DataFrame(columns=['time', 'x', 'y', 'z', 'temperature'])  # empty df

    def _parse_single_page(self, page):
        """ Parse content of a single data page. """
        page_index, page_str = page
        data_block = ''
        try:
            lines = page_str.splitlines()
            header_lines = []
            header_prefixes = ['sequence number:', 'device:', 'time:', 'page time:', 'temperature:',
                               'battery voltage:', 'device status:', 'measurement frequency:',
                               'light:', 'button:', 'waveform:', 'unassigned:']
            for line in lines:
                line_lower = line.lower()
                if any(line_lower.startswith(prefix) for prefix in header_prefixes):
                    header_lines.append(line)
                else:
                    data_block = line.strip()  # assuming the first non-header line is the data line
                    break  # exit after finding first data line

            # extract the page header information
            block_time_str = ''
            temperature, freq, battery, light = None, None, None, None
            for header_line in header_lines:
                header_line_lower = header_line.lower()
                if 'page time:' in header_line_lower:
                    block_time_str = header_line.split(':', 1)[1].strip()
                elif 'time:' in header_line_lower:
                    block_time_str = header_line.split(':', 1)[1].strip()
                elif 'temperature:' in header_line_lower:
                    temperature = float(header_line.split(':', 1)[1].strip())
                elif 'measurement frequency:' in header_line_lower:
                    freq = float(header_line.split(':', 1)[1].strip())
                elif 'battery voltage:' in header_line_lower:
                    battery = float(header_line.split(':', 1)[1].strip())

            if not data_block:  # check if data line is empty
                logger.warning(f"No data line found in page {page_index}. Skipping page.")
                self.skipped_pages.append(page_index)
                return None

            # Parse block time and data line and add page index to df
            block_time = self._format_page_time(block_time_str)
            block_df = self._parse_data_block(data_block, block_time, freq)
            block_df['page_index'] = page_index
            if self.kwargs.get('incl_temp'):
                block_df['temperature'] = temperature
            if self.kwargs.get('incl_battery'):
                block_df['battery'] = battery

            return block_df

        except Exception as e:
            self.failed_pages.append(page_index)
            logger.error(f"Error processing page {page_index}: {e}\nData line: {data_block}")
            return None

    def _parse_data_block(self, data_block: str, page_time: int, freq: float):
        """ Parse a data line and extract accelerometer samples.

        Parameters:
            :param data_block: (str) The line containing hexadecimal data.
            :param page_time: (str) The starting timestamp of the page in ms.
            :param freq: (float) The page sampling frequency in Hz.

        Return:
            :return: (pd.DataFrame) containing the actimeter data of the pages' data block.
        """
        data_block = data_block.strip()  # remove any whitespace

        # validate that the data block contains only valid hex characters
        if not data_block or not all(c in '0123456789ABCDEFabcdef' for c in data_block):
            raise ValueError(f"Data line contains invalid characters: {data_block}")

        # each sample: 6 bytes where 1 byte is represented by 2 hex chars by definition -> 1 sample: 12 hex chars!
        if len(data_block) % 12 != 0:
            raise ValueError(f"Data block length is not a multiple of 12: {len(data_block)}")

        num_samples = len(data_block) // 12  # each sample is 12 hex characters
        if num_samples == 0:
            return pd.DataFrame()

        # extract hex values
        x_hex = [data_block[i * 12:i * 12 + 3] for i in range(num_samples)]  # 1st 1.5bytes = 12 bits = 3 hex chars
        y_hex = [data_block[i * 12 + 3:i * 12 + 6] for i in range(num_samples)]  # 2nd 3 hex chars
        z_hex = [data_block[i * 12 + 6:i * 12 + 9] for i in range(num_samples)]  # 3rd 3 hex chars

        try:  # convert hex to integers
            x_raw = np.array([self._12bit_hex_to_decimal(h, mode='xyz') for h in x_hex], dtype=np.int32)
            y_raw = np.array([self._12bit_hex_to_decimal(h, mode='xyz') for h in y_hex], dtype=np.int32)
            z_raw = np.array([self._12bit_hex_to_decimal(h, mode='xyz') for h in z_hex], dtype=np.int32)

        except ValueError as e:
            raise ValueError(f"Error converting hex to int: {e}")

        # calibrate the raw values
        if (self.kwargs.get('calibrate_with_gain_and_offset')
                and self.mfr_gain is not None and self.mfr_offset is not None):
            x = (x_raw * 100.0 - self.mfr_offset[0]) / self.mfr_gain[0]
            y = (y_raw * 100.0 - self.mfr_offset[1]) / self.mfr_gain[1]
            z = (z_raw * 100.0 - self.mfr_offset[2]) / self.mfr_gain[2]
        else:
            x, y, z = x_raw, y_raw, z_raw

        sample_indices = np.arange(num_samples)  # generate sample indices (to maintain ordering in parallel processing)
        timestamps = page_time + sample_indices * (1.0 / freq) * 1000  # unix milliseconds
        # note: GENEActiv only stores timestamp + freq per page, i.e. timestamps have to be generated for each page
        # (so strictly speaking resampling)

        data_dict = {'time': pd.to_datetime(timestamps, unit='ms'),
                     'x': x, 'y': y, 'z': z,
                     'sample_index': sample_indices}

        if self.kwargs.get('incl_light'):
            light_and_button_hex = [data_block[i * 12 + 9:i * 12 + 12] for i in range(num_samples)]
            try:
                light_raw = np.array(  # 'light' mode will extraxt 10 first bits (unsigned)
                    [self._12bit_hex_to_decimal(h, mode='light') for h in light_and_button_hex], dtype=np.int32)
            except ValueError as e:
                raise ValueError(f"Error converting light hex to int: {e}")
            data_dict.update({'light': light_raw * self.light_lux / self.light_volts})

        if self.kwargs.get('incl_button'):
            light_and_button_hex = [data_block[i * 12 + 9:i * 12 + 12] for i in range(num_samples)]
            try:
                button = np.array(  # 'button' mode will extraxt 11th bit
                    [self._12bit_hex_to_decimal(h, mode='button') for h in light_and_button_hex], dtype=np.int32)
            except ValueError as e:
                raise ValueError(f"Error converting light hex to int: {e}")
            data_dict.update({'button': button})

        return pd.DataFrame(data_dict)

    @staticmethod
    def _format_page_time(time_str):
        """ Format the page time string to timestamp in ms.
        Parameters:
            :param time_str: (str) the time string extracted from the page header.
        Returns:
            :return: (dt.timestamp) in milliseconds. """
        time_formats = ['%Y-%m-%d %H:%M:%S:%f', '%Y-%m-%d %H:%M:%S']
        for fmt in time_formats:
            try:
                dt = datetime.datetime.strptime(time_str, fmt)
                timestamp = int(dt.timestamp() * 1000)
                return timestamp
            except ValueError:
                continue
        raise ValueError(f'Invalid time format: {time_str}')

    @staticmethod
    def _12bit_hex_to_decimal(hex_str: str, mode: str = 'xyz'):
        """ Convert the 12-bit hex value (3 hex chars) to decimal based on different modes.
        Parameters:
            :param hex_str: (str) the str of 3 hex chars to convert.
            :param mode: (str, Optional) 'xyz' for full signed 12 bits, i.e., ±2048 (default),
             'light' for the unsigned first 10 bits (0-1024) and 'button' for the second last bit (0/1).
        Returns:
            :return: the decimal value. """
        value = int(hex_str, 16)  # convert hex to int
        if mode == 'xyz':  # use all 12 bits and apply sign
            if value >= 2048:  # check if the value is negative in 12-bit signed integer range
                value -= 4096  # adjust for overflow in signed 12-bit range
            return value

        elif mode == 'light':  # use first 10 bits (no sign)
            value = value >> 2  # shift right by 2 to remove the last two bits
            return value

        elif mode == 'button':  # extract second last bit (11th position) (0 or 1)
            return (value >> 1) & 0x1  # shift right by 1 and mask to get only the second last bit

        else:
            raise ValueError(f"Invalid mode: {mode}. Choose from 'xyz', 'light', 'button', or 'test'.")
