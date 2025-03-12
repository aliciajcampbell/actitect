import argparse
import gc
import logging
import traceback
from itertools import chain
from pathlib import Path

import pandas as pd

import aktiRBD.utils as utils
from aktiRBD.actimeter import SUPPORTED_FILETYPES
from aktiRBD.processing.file_processor import FileProcessor


def _process_dataset(args: argparse.Namespace, processing_kwargs: dict, feature_kwargs: dict, logger: logging.Logger):
    _data_dir = Path(args.root_dir).joinpath(args.data_dir)
    _meta_file = Path(args.root_dir).joinpath(args.meta_file)
    assert _data_dir.is_dir(), f"the passed 'data_dir' must exist and be a directory! ({_data_dir})."
    assert _meta_file.is_file(), f"can't find 'metadata.csv' file {_meta_file}."

    meta_df = utils.read_meta_csv_to_df(_meta_file)
    _raw_files = sorted(chain.from_iterable(_data_dir.glob(f"*{ext}") for ext in SUPPORTED_FILETYPES))

    if not _raw_files:
        raise UserWarning(f"No actigraphy files found at {_data_dir}")
    save_dir = utils.check_make_dir(Path(args.root_dir).joinpath(args.out_dir), use_existing=True)

    try:
        sleep_logs = utils.standardize_sleep_diary(pd.read_excel(
            _data_dir.joinpath('meta/RBD_HC-RBD_Bettzeiten_20240117_fixed.xlsx'))) if args.create_plots else None
    except FileNotFoundError:
        sleep_logs = None
        logger.warning(f"no sleep log file found.")

    _excluded = {}
    with utils.custom_tqdm(total=len(_raw_files), position=0, leave=True) as pbar:
        for i, file_path in enumerate(_raw_files):

            try:
                _meta = meta_df[meta_df['filename'] == file_path.name]
                if _meta.empty or _meta.shape[0] > 1:
                    raise UserWarning(f"missing or double entry in 'data/raw/meta/metadata.csv' for {file_path},"
                                      f" fix and try again")
                _meta = _meta.iloc[0].to_dict()
                _patient_id, _diagnosis = _meta.get('ID') or 'none', _meta.get('diagnosis') or 'none'
                if _meta['exclude'] == 1:
                    _excluded.update({_patient_id: "meta"})
                    raise UserWarning(f"excluding {_patient_id} according to meta.")
                else:
                    sleep_log = sleep_logs[sleep_logs['ID'] == _patient_id] \
                        if args.create_plots and sleep_logs is not None else None

                    file_processor = FileProcessor(_patient_id, _diagnosis, file_path, save_dir,
                                                   save_processed_data=args.save_processed, sleep_log=sleep_log)
                    file_processor.process(feature_kwargs, processing_kwargs, args, pbar)
                    del file_processor
                    gc.collect()

            except NotImplementedError as _ne:
                logger.warning(f" NotImplementError {_ne} encountered in process_single_file({_patient_id})."
                               f" Might be caused by unknown 'feature_mode' variable: "
                               f"Got '{feature_kwargs['feature_mode']}, currently implemented: 'per_night'")
                logger.warning(f" Excluded: {_patient_id}")
                _excluded.update({_patient_id: f"{_ne}"})
            except UserWarning as _uw:
                logger.warning(f" UserWarning {_uw} encountered in process_single_file({_patient_id}).")
                logger.warning(f" Excluded: {_patient_id}")
                _excluded.update({_patient_id: f"{_uw}"})
            except Exception as e:
                logger.warning(f" Exception {e} encountered in process_single_file({_patient_id}).")
                logger.warning(f" Excluded: {_patient_id}")
                _excluded.update({_patient_id: f"{e}"})
                traceback.print_exc()
            pbar.update(1)

        logger.info(f"\n ...processing finished. Excluded: {_excluded} -> n_included={len(_raw_files)}-{len(_excluded)}"
                    f"={len(_raw_files) - len(_excluded)}")


def _parse_args():
    """ Parses command-line arguments for preprocessing. Use 'aktiRBD_preprocess --help' for details."""
    parser = argparse.ArgumentParser(
        prog='aktiRBD-preprocess',
        description='Preprocess actigraphy data by standardizing, segmenting sleep windows,'
                    ' and calculating nocturnal motion features.',
        formatter_class=argparse.RawTextHelpFormatter)

    # file organization arguments
    parser.add_argument(
        '-r', '--root_dir', type=str,
        default=str(utils.get_experiment_root()),  # default points to aktiRBD_experiment
        help="The root directory containing 'data/' and 'aktiRBD/'."
    )

    parser.add_argument(
        '-c', '--config_file', type=str,
        default='./aktiRBD/src/aktiRBD/config/preprocessing.yaml',
        help='Location of the config .yaml file that defines preprocessing settings.'
    )

    parser.add_argument(
        '-d', '--data_dir', type=str,
        default='./data/raw/',
        help="The directory containing the raw actigraphy files, relative to --root_dir."
    )

    parser.add_argument(
        '-m', '--meta_file', type=str,
        default='./data/raw/meta/metadata.csv',
        help="The path to the metadata.csv, relative to --root_dir."
    )

    parser.add_argument(
        '-o', '--out_dir', type=str,
        default='./data/processed/',
        help="The directory used to store the features of each subject, relative to --root_dir."
    )

    # operational arguments
    parser.add_argument(
        '-ns', '--no-store', dest='save_processed',
        action='store_false', default=True,
        help='If used, will only save the calculated features, not the processed data.'
    )

    parser.add_argument(
        '-cp', '--create_plots', action='store_true', default=True,
        help='Generate plots during processing or using existing processed data. (default: True).'
    )

    parser.add_argument(
        '-rp', '--redo_processing', action='store_true', default=False,
        help='Force reprocessing of raw data even if preprocessed files exist (default: False).'
    )

    parser.add_argument(
        '-sf', '--skip_feature_calc', action='store_true', default=False,
        help='Skip feature calculation (default: False).'
    )

    parser.add_argument(
        '-del', '--delete_processed_files', action='store_true', default=False,
        help='If used, will delete previously processed .parquet files to free up space. Requires confirmation.'
    )

    return parser.parse_args()


def main():

    args = _parse_args()
    log_path = utils.check_make_dir(Path(args.root_dir).joinpath(f"{args.data_dir}/logs/"), True)
    utils.setup_logging(log_file_path=log_path.joinpath('preprocess.log'))
    logger = logging.getLogger(__name__)

    config_params = utils.load_yaml_file(Path(args.root_dir).joinpath(args.config_file))
    config_params.update({'args': args})

    _process_dataset(**config_params, logger=logger)


if __name__ == '__main__':
    main()
