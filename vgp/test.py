import _init_paths
import os
import argparse
from copy import deepcopy

from vgp.function.config import config, update_config
from vgp.function.test import test_net


def parse_args():
    parser = argparse.ArgumentParser('Get Jointly Test Result of Cognition Network')
    parser.add_argument('--cfg', type=str, help='path to config yaml')
    parser.add_argument('--ckpt', type=str, help='path to checkpoint of trained net')
    parser.add_argument('--bs', type=int)
    parser.add_argument('--gpus', type=int, nargs='+', default=[0])
    parser.add_argument('--result-path', type=str, help='path to store test result csv file.', default='./test_result')
    parser.add_argument('--result-name', type=str)
    parser.add_argument('--fp16', default=False, action='store_true')
    parser.add_argument('-pred', '--repredict', default=False, action='store_true')

    args = parser.parse_args()

    if args.cfg is not None:
        config.update(deepcopy(args.cfg))
    if args.bs is not None:
        config.TEST.BATCH_IMAGES = args.bs

    return args, config


def main():
    args, config = parse_args()
    test_net(args, config, ckpt_path=args.ckpt, save_path=args.result_path, save_name=args.result_name)


if __name__ == '__main__':
    main()


