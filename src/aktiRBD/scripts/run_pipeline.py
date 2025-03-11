import argparse
import datetime
import logging
import sys
from pathlib import Path

from aktiRBD import __file__ as _aktirbd_root_file
from aktiRBD import utils
from aktiRBD.classifier.tools.pipeline import TrainPipeline, TestPipeline
from aktiRBD.config import PipelineConfig, ExternalTestConfig

logger = logging.getLogger(__name__)


def _run_setup():
    def _parse_args():
        parser = argparse.ArgumentParser(
            prog='aktiRBD',
            description='Entry point to perform RBD analysis from actigraphy data. Call --help or -h for details.',
            formatter_class=argparse.RawTextHelpFormatter)

        # arguments that define the operation mode of the script:
        group = parser.add_mutually_exclusive_group(required=False)  # to enable default
        group.add_argument('--train_eval', action='store_true', help='Run the entire training pipeline on '
                                                                     'cologne data: nested cv on train set, train model'
                                                                     ' on entire train set, eval on test set. ')
        group.add_argument('--test', action='store_true', help='Use pretrained models to test on unseen data. (Default)')
        group.add_argument('--predict', action='store_true', help='Only run inference, no testing.')

        # Additional arguments for options within modes
        parser.add_argument('-r', '--root_dir', type=str, default=str(Path(__file__).resolve().parents[4]),
                            help="The root directory containing 'data/' and 'aktiRBD/'.")

        parser.add_argument('-c', '--config_file', type=str,
                            default='./aktiRBD/src/aktiRBD/config/external_test.yaml',  # rel. to root dir
                            help='full path (rel. to root) of the config .yaml file defining preprocessing settings.')

        parser.add_argument('-d', '--processed_data_dir', type=str,
                            default='./data/processed/',  # rel. to root dir
                            help='directory (rel. to root) containing the pre-processed data.')

        parser.add_argument('-m', '--meta_file', type=str,
                            default='./data/raw/meta/metadata.csv',  # rel. to root dir
                            help="meta data file path (rel. to root) that contains list of patients. Has to contain "
                                 "the class labels except for execution in 'predict' mode. ")

        _args = parser.parse_args()

        # Set default operation mode:
        if not any([_args.train_eval, _args.test, _args.predict]):
            _args.test = True

        # cast path-like args into PosixPath types
        _args.root_dir = Path(_args.root_dir)
        _args.config_file = _args.root_dir.joinpath(_args.config_file)
        _args.processed_data_dir = _args.root_dir.joinpath(_args.processed_data_dir)
        _args.meta_file = _args.root_dir.joinpath(_args.meta_file)

        assert (_args.root_dir.is_dir() and _args.root_dir.joinpath('data').is_dir() and
                _args.root_dir.joinpath('aktiRBD').is_dir()), \
            (f" '--root_dir' ('-r') ({_args.root_dir}) is not a dir "
             f"or does not contain sub-dirs './data' and './aktiRBD'")
        assert _args.config_file.is_file(), \
            f" '--config_file' (-'c') is not a file. ({_args.config_file})"

        return _args

    args = _parse_args()
    _experiment_id = datetime.datetime.now().strftime('%Y-%m-%d_%Hh%Mm%Ss')
    root_save_path = utils.check_make_dir(Path(args.root_dir).joinpath(f"results/pipeline/run_{_experiment_id}"))
    utils.setup_logging(log_file_path=root_save_path.joinpath('log'))

    if args.train_eval:
        config = PipelineConfig.from_yaml(args.config_file)
    elif args.test or args.predict:
        config = ExternalTestConfig.from_yaml(args.config_file)
        config.pretrained_model_dirs = [args.root_dir.joinpath(_dir) for _dir in config.pretrained_model_dirs]
    else:
        raise ValueError(f"must choose one of args.train_eval, args.test or args.predict.")

    _case_ident = f"{config.data.loader.feature_dir.replace('/features', '')}" + "_" + "_".join(
        config.data.loader.aggregation) + f"/{sys.platform}" + f"/scale_{config.data.processing.scaling_order}"
    if config.data.processing.scaling_order == 'before_ranking':
        _case_ident += f"/{config.data.processing.scaler}_scaler"

    if args.train_eval:
        if not config.nested_cv.load_path_cv_feature_rankings:
            config.nested_cv.load_path_cv_feature_rankings = args.root_dir.joinpath(
                f"results/feat_rank/cv/{_case_ident}")
        else:
            config.nested_cv.load_path_cv_feature_rankings = Path(config.nested_cv.load_path_cv_feature_rankings)

        if not config.final_model.load_path_feature_rankings:
            config.final_model.load_path_feature_rankings = args.root_dir.joinpath(
                f"results/feat_rank/full/{_case_ident}")
        else:
            config.final_model.load_path_feature_rankings = Path(config.final_model.load_path_feature_rankings)

        if not config.final_model.nested_cv_path:
            config.final_model.nested_cv_path = root_save_path.joinpath('nested_cv')
        else:
            config.final_model.nested_cv_path = Path(config.final_model.nested_cv_path)

        if not config.final_model.save_path_models:
            config.final_model.save_path_models = root_save_path.joinpath('models')
            if config.final_model.overwrite_final_repo_models:
                config.final_model.save_path_models = [config.final_model.save_path_models]
                config.final_model.save_path_models.append(
                    Path(_aktirbd_root_file).parent.joinpath(f"models/pretrained/"))
        else:
            config.final_model.save_path_models = Path(config.final_model.save_path_models)

    # save the settings
    _dump_dict = {'command line options': vars(args), 'config': config.dict()}
    utils.dump_to_json(_dump_dict, root_save_path.joinpath('settings.json'))

    return args, config, root_save_path


def main():
    # setup
    args, config, save_path = _run_setup()

    # define and run pipeline
    pipeline_cls = TrainPipeline if args.train_eval else TestPipeline
    pipeline = pipeline_cls(args, config, save_path)
    pipeline.run()


if __name__ == '__main__':
    main()
