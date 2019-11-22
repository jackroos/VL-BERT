import _init_paths
import os
import argparse
from copy import deepcopy

from vqa.function.config import config, update_config
from vqa.function.test import test_net


def parse_args():
    parser = argparse.ArgumentParser('Get Test Result of VQA Network')
    parser.add_argument('--cfg', type=str, help='path to answer net config yaml')
    parser.add_argument('--ckpt', type=str, help='path to checkpoint of answer net')
    parser.add_argument('--bs', type=int)
    parser.add_argument('--gpus', type=int, nargs='+')
    parser.add_argument('--model-dir', type=str, help='root path to store checkpoint')
    parser.add_argument('--result-path', type=str, help='path to store test result file.')
    parser.add_argument('--result-name', type=str)
    parser.add_argument('--split', default='test2015')

    args = parser.parse_args()

    if args.cfg is not None:
        update_config(args.cfg)
    if args.bs is not None:
        config.TEST.BATCH_IMAGES = args.bs
    if args.gpus is not None:
        config.GPUS = ','.join([str(gpu) for gpu in args.gpus])
    if args.split is not None:
        config.DATASET.TEST_IMAGE_SET = args.split
    if args.model_dir is not None:
        config.OUTPUT_PATH = os.path.join(args.model_dir, config.OUTPUT_PATH)

    return args, config


def main():
    args, config = parse_args()

    result_json_path = test_net(args, config,
                                ckpt_path=args.ckpt, save_path=args.result_path, save_name=args.result_name)


if __name__ == '__main__':
    main()
