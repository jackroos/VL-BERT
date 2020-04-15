import os
import pprint
import shutil
import inspect

from tensorboardX import SummaryWriter
import numpy as np
import torch
import torch.nn
import torch.optim as optim
import torch.distributed as distributed
from torch.nn.parallel import DistributedDataParallel as DDP

from common.utils.create_logger import create_logger
from common.utils.misc import summary_parameters, bn_fp16_half_eval
from common.utils.load import smart_resume, smart_partial_load_model_state_dict
from common.trainer import train
from common.metrics.composite_eval_metric import CompositeEvalMetric
from common.metrics import vcr_metrics
from common.callbacks.batch_end_callbacks.speedometer import Speedometer
from common.callbacks.epoch_end_callbacks.validation_monitor import ValidationMonitor
from common.callbacks.epoch_end_callbacks.checkpoint import Checkpoint
from common.lr_scheduler import WarmupMultiStepLR
from common.nlp.bert.optimization import AdamW, WarmupLinearSchedule
from vcr.data.build import make_dataloader, build_dataset, build_transforms
from vcr.modules import *
from vcr.function.val import do_validation
import yaml

with open("/Users/jules-samaran/PycharmProjects/VL-BERT/cfgs/vcr/base_q2a_4x16G_fp32.yaml", "r") as file:
    config = yaml.load(file)
    model = eval(config.MODULE)(config)

    train_loader = make_dataloader(config, mode='train', distributed=False)

    model.train()
    # training
    for nbatch, batch in enumerate(train_loader):
        outputs, loss = model(*batch)
