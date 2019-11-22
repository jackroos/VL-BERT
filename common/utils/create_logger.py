# --------------------------------------------------------
# Deformable Convolutional Networks
# Copyright (c) 2017 Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Bin Xiao
# --------------------------------------------------------

import os
import logging
import time
import errno


def makedirsExist(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno == errno.EEXIST:
            print('Directory not created.')
        else:
            raise


def create_logger(root_output_path, config_file, image_set, split='train', hypers=()):
    # set up logger
    if not os.path.exists(root_output_path):
        makedirsExist(root_output_path)
    assert os.path.exists(root_output_path), '{} does not exist'.format(root_output_path)

    cfg_name = os.path.splitext(os.path.basename(config_file))[0]

    config_output_path = os.path.join(root_output_path, '{}'.format(cfg_name))
    for (hyper_name, hyper_val) in hypers:
        config_output_path += '@{}={}'.format(hyper_name, hyper_val)
    if not os.path.exists(config_output_path):
        makedirsExist(config_output_path)

    final_output_path = os.path.join(config_output_path, image_set + '_' + split)
    if not os.path.exists(final_output_path):
        makedirsExist(final_output_path)

    log_file = '{}_{}.log'.format(cfg_name, time.strftime('%Y-%m-%d-%H-%M'))
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=os.path.join(final_output_path, log_file), format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    return logger, final_output_path


