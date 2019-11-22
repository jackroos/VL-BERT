import _init_paths
import os
import argparse
from copy import deepcopy

from vcr.function.config import config, update_config
from vcr.function.test import test_net, merge_result


def parse_args():
    parser = argparse.ArgumentParser('Get Jointly Test Result of Cognition Network')
    parser.add_argument('--a-cfg', type=str, help='path to answer net config yaml')
    parser.add_argument('--r-cfg', type=str, help='path to rationale net config yaml')
    parser.add_argument('--a-ckpt', type=str, help='path to checkpoint of answer net')
    parser.add_argument('--r-ckpt', type=str, help='path to checkpoint of rationale net')
    parser.add_argument('--a-bs', type=int)
    parser.add_argument('--r-bs', type=int)
    parser.add_argument('--gpus', type=int, nargs='+', default=[0])
    parser.add_argument('--test-file', type=str)
    parser.add_argument('--result-path', type=str, help='path to store test result csv file.', default='./test_result')
    parser.add_argument('--result-name', type=str)
    parser.add_argument('--fp16', default=False, action='store_true')
    parser.add_argument('--use-cache', default=False, action='store_true')

    args = parser.parse_args()
    a_config = r_config = None
    reset_config = deepcopy(config)
    if args.a_cfg is not None:
        a_config = config
        if reset_config is not None:
            a_config.update(deepcopy(reset_config))
        if args.a_cfg is not None:
            update_config(args.a_cfg)
        a_config = deepcopy(a_config)
    if args.r_cfg is not None:
        r_config = config
        if reset_config is not None:
            r_config.update(deepcopy(reset_config))
        if args.r_cfg is not None:
            update_config(args.r_cfg)
        r_config = deepcopy(r_config)
    if args.a_bs is not None:
        a_config.TEST.BATCH_IMAGES = args.a_bs
    if args.r_bs is not None:
        r_config.TEST.BATCH_IMAGES = args.r_bs

    if args.test_file is not None:
        a_config.DATASET.TEST_ANNOTATION_FILE = args.test_file
        r_config.DATASET.TEST_ANNOTATION_FILE = args.test_file

    return args, a_config, r_config


def main():
    args, a_config, r_config = parse_args()

    if args.a_ckpt:
        a_config.DATASET.TASK = 'Q2A'
        a_config.GPUS = ','.join([str(k) for k in args.gpus])
        a_result_csv = test_net(args,
                                a_config,
                                ckpt_path=args.a_ckpt,
                                save_path=args.result_path,
                                save_name=args.result_name)
    if args.r_ckpt:
        r_config.DATASET.TASK = 'QA2R'
        r_config.GPUS = ','.join([str(k) for k in args.gpus])
        r_result_csv = test_net(args,
                                r_config,
                                ckpt_path=args.r_ckpt,
                                save_path=args.result_path,
                                save_name=args.result_name)
    if args.a_ckpt and args.r_ckpt:
        merge_result(a_result_csv, r_result_csv,
                     os.path.join(args.result_path, '{}_test_result_Q2AR.csv'.format(args.result_name)))


if __name__ == '__main__':
    main()


