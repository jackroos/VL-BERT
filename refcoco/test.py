import _init_paths
import os
import argparse

from refcoco.function.config import config, update_config
from refcoco.function.test import test_net


def parse_args():
    parser = argparse.ArgumentParser('Train Cognition Network')
    parser.add_argument('--cfg', type=str, help='path to config file')
    parser.add_argument('--ckpt', type=str, help='root path to store checkpoint')
    parser.add_argument('--gpus', type=int, nargs='+', help='indices of GPUs to use', default=[0])
    parser.add_argument('--bs', type=int)
    parser.add_argument('--split', type=str, choices=['test', 'testA', 'testB', 'val'], default='val')
    parser.add_argument('--result-path', type=str, help='dir to save result file')
    parser.add_argument('--result-name', type=str, help='name of result file')

    args = parser.parse_args()

    if args.cfg is not None:
        update_config(args.cfg)

    config.GPUS = ','.join([str(index) for index in args.gpus])

    if args.bs is not None:
        config.TEST.BATCH_IMAGES = args.bs

    return args, config


def main():
    args, config = parse_args()

    test_net(args, config)


if __name__ == '__main__':
    main()
