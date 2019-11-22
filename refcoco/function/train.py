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
from common.metrics import refcoco_metrics
from common.callbacks.batch_end_callbacks.speedometer import Speedometer
from common.callbacks.epoch_end_callbacks.validation_monitor import ValidationMonitor
from common.callbacks.epoch_end_callbacks.checkpoint import Checkpoint
from common.lr_scheduler import WarmupMultiStepLR
from common.nlp.bert.optimization import AdamW, WarmupLinearSchedule
from refcoco.data.build import make_dataloader, build_dataset, build_transforms
from refcoco.modules import *
from refcoco.function.val import do_validation

try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as Apex_DDP
except ImportError:
    pass
    #raise ImportError("Please install apex from https://www.github.com/nvidia/apex if you want to use fp16.")


def train_net(args, config):
    # setup logger
    logger, final_output_path = create_logger(config.OUTPUT_PATH, args.cfg, config.DATASET.TRAIN_IMAGE_SET,
                                              split='train')
    model_prefix = os.path.join(final_output_path, config.MODEL_PREFIX)
    if args.log_dir is None:
        args.log_dir = os.path.join(final_output_path, 'tensorboard_logs')

    pprint.pprint(args)
    logger.info('training args:{}\n'.format(args))
    pprint.pprint(config)
    logger.info('training config:{}\n'.format(pprint.pformat(config)))

    # manually set random seed
    if config.RNG_SEED > -1:
        np.random.seed(config.RNG_SEED)
        torch.random.manual_seed(config.RNG_SEED)
        torch.cuda.manual_seed_all(config.RNG_SEED)

    # cudnn
    torch.backends.cudnn.benchmark = False
    if args.cudnn_off:
        torch.backends.cudnn.enabled = False

    if args.dist:
        model = eval(config.MODULE)(config)
        local_rank = int(os.environ.get('LOCAL_RANK') or 0)
        config.GPUS = str(local_rank)
        torch.cuda.set_device(local_rank)
        master_address = os.environ['MASTER_ADDR']
        master_port = int(os.environ['MASTER_PORT'] or 23456)
        world_size = int(os.environ['WORLD_SIZE'] or 1)
        rank = int(os.environ['RANK'] or 0)
        if args.slurm:
            distributed.init_process_group(backend='nccl')
        else:
            distributed.init_process_group(
                backend='nccl',
                init_method='tcp://{}:{}'.format(master_address, master_port),
                world_size=world_size,
                rank=rank,
                group_name='mtorch')
        print(f'native distributed, size: {world_size}, rank: {rank}, local rank: {local_rank}')
        torch.cuda.set_device(local_rank)
        config.GPUS = str(local_rank)
        model = model.cuda()
        if not config.TRAIN.FP16:
            model = DDP(model, device_ids=[local_rank], output_device=local_rank)

        if rank == 0:
            summary_parameters(model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model,
                               logger)
            shutil.copy(args.cfg, final_output_path)
            shutil.copy(inspect.getfile(eval(config.MODULE)), final_output_path)

        writer = None
        if args.log_dir is not None:
            tb_log_dir = os.path.join(args.log_dir, 'rank{}'.format(rank))
            if not os.path.exists(tb_log_dir):
                os.makedirs(tb_log_dir)
            writer = SummaryWriter(log_dir=tb_log_dir)

        train_loader, train_sampler = make_dataloader(config,
                                                      mode='train',
                                                      distributed=True,
                                                      num_replicas=world_size,
                                                      rank=rank,
                                                      expose_sampler=True)
        val_loader = make_dataloader(config,
                                     mode='val',
                                     distributed=True,
                                     num_replicas=world_size,
                                     rank=rank)

        batch_size = world_size * (sum(config.TRAIN.BATCH_IMAGES)
                                   if isinstance(config.TRAIN.BATCH_IMAGES, list)
                                   else config.TRAIN.BATCH_IMAGES)
        if config.TRAIN.GRAD_ACCUMULATE_STEPS > 1:
            batch_size = batch_size * config.TRAIN.GRAD_ACCUMULATE_STEPS
        base_lr = config.TRAIN.LR * batch_size
        optimizer_grouped_parameters = [{'params': [p for n, p in model.named_parameters() if _k in n],
                                         'lr': base_lr * _lr_mult}
                                        for _k, _lr_mult in config.TRAIN.LR_MULT]
        optimizer_grouped_parameters.append({'params': [p for n, p in model.named_parameters()
                                                        if all([_k not in n for _k, _ in config.TRAIN.LR_MULT])]})
        if config.TRAIN.OPTIMIZER == 'SGD':
            optimizer = optim.SGD(optimizer_grouped_parameters,
                                  lr=config.TRAIN.LR * batch_size,
                                  momentum=config.TRAIN.MOMENTUM,
                                  weight_decay=config.TRAIN.WD)
        elif config.TRAIN.OPTIMIZER == 'Adam':
            optimizer = optim.Adam(optimizer_grouped_parameters,
                                   lr=config.TRAIN.LR * batch_size,
                                   weight_decay=config.TRAIN.WD)
        elif config.TRAIN.OPTIMIZER == 'AdamW':
            optimizer = AdamW(optimizer_grouped_parameters,
                              lr=config.TRAIN.LR * batch_size,
                              betas=(0.9, 0.999),
                              eps=1e-6,
                              weight_decay=config.TRAIN.WD,
                              correct_bias=True)
        else:
            raise ValueError('Not support optimizer {}!'.format(config.TRAIN.OPTIMIZER))
        total_gpus = world_size

    else:
        #os.environ['CUDA_VISIBLE_DEVICES'] = config.GPUS
        model = eval(config.MODULE)(config)
        summary_parameters(model, logger)
        shutil.copy(args.cfg, final_output_path)
        shutil.copy(inspect.getfile(eval(config.MODULE)), final_output_path)
        num_gpus = len(config.GPUS.split(','))
        assert num_gpus <= 1 or (not config.TRAIN.FP16), "Not support fp16 with torch.nn.DataParallel. " \
                                                         "Please use amp.parallel.DistributedDataParallel instead."
        total_gpus = num_gpus
        rank = None
        writer = SummaryWriter(log_dir=args.log_dir) if args.log_dir is not None else None

        # model
        if num_gpus > 1:
            model = torch.nn.DataParallel(model, device_ids=[int(d) for d in config.GPUS.split(',')]).cuda()
        else:
            torch.cuda.set_device(int(config.GPUS))
            model.cuda()

        # loader
        train_loader = make_dataloader(config, mode='train', distributed=False)
        val_loader = make_dataloader(config, mode='val', distributed=False)
        train_sampler = None

        batch_size = num_gpus * (sum(config.TRAIN.BATCH_IMAGES) if isinstance(config.TRAIN.BATCH_IMAGES, list)
                                 else config.TRAIN.BATCH_IMAGES)
        if config.TRAIN.GRAD_ACCUMULATE_STEPS > 1:
            batch_size = batch_size * config.TRAIN.GRAD_ACCUMULATE_STEPS
        base_lr = config.TRAIN.LR * batch_size
        optimizer_grouped_parameters = [{'params': [p for n, p in model.named_parameters() if _k in n],
                                         'lr': base_lr * _lr_mult}
                                        for _k, _lr_mult in config.TRAIN.LR_MULT]
        optimizer_grouped_parameters.append({'params': [p for n, p in model.named_parameters()
                                                        if all([_k not in n for _k, _ in config.TRAIN.LR_MULT])]})

        if config.TRAIN.OPTIMIZER == 'SGD':
            optimizer = optim.SGD(optimizer_grouped_parameters,
                                  lr=config.TRAIN.LR * batch_size,
                                  momentum=config.TRAIN.MOMENTUM,
                                  weight_decay=config.TRAIN.WD)
        elif config.TRAIN.OPTIMIZER == 'Adam':
            optimizer = optim.Adam(optimizer_grouped_parameters,
                                   lr=config.TRAIN.LR * batch_size,
                                   weight_decay=config.TRAIN.WD)
        elif config.TRAIN.OPTIMIZER == 'AdamW':
            optimizer = AdamW(optimizer_grouped_parameters,
                              lr=config.TRAIN.LR * batch_size,
                              betas=(0.9, 0.999),
                              eps=1e-6,
                              weight_decay=config.TRAIN.WD,
                              correct_bias=True)
        else:
            raise ValueError('Not support optimizer {}!'.format(config.TRAIN.OPTIMIZER))

    # partial load pretrain state dict
    if config.NETWORK.PARTIAL_PRETRAIN != "":
        pretrain_state_dict = torch.load(config.NETWORK.PARTIAL_PRETRAIN, map_location=lambda storage, loc: storage)['state_dict']
        prefix_change = [prefix_change.split('->') for prefix_change in config.NETWORK.PARTIAL_PRETRAIN_PREFIX_CHANGES]
        if len(prefix_change) > 0:
            pretrain_state_dict_parsed = {}
            for k, v in pretrain_state_dict.items():
                no_match = True
                for pretrain_prefix, new_prefix in prefix_change:
                    if k.startswith(pretrain_prefix):
                        k = new_prefix + k[len(pretrain_prefix):]
                        pretrain_state_dict_parsed[k] = v
                        no_match = False
                        break
                if no_match:
                    pretrain_state_dict_parsed[k] = v
            pretrain_state_dict = pretrain_state_dict_parsed
        smart_partial_load_model_state_dict(model, pretrain_state_dict)

    # metrics
    train_metrics_list = [refcoco_metrics.RefAccuracy(allreduce=args.dist,
                                                      num_replicas=world_size if args.dist else 1),
                          refcoco_metrics.ClsAccuracy(allreduce=args.dist,
                                                      num_replicas=world_size if args.dist else 1),
                          refcoco_metrics.ClsPosAccuracy(allreduce=args.dist,
                                                         num_replicas=world_size if args.dist else 1),
                          refcoco_metrics.ClsPosFraction(allreduce=args.dist,
                                                         num_replicas=world_size if args.dist else 1),
                          ]
    val_metrics_list = [refcoco_metrics.RefAccuracy(allreduce=args.dist,
                                                    num_replicas=world_size if args.dist else 1)]
    for output_name, display_name in config.TRAIN.LOSS_LOGGERS:
        train_metrics_list.append(
            refcoco_metrics.LossLogger(output_name, display_name=display_name, allreduce=args.dist,
                                       num_replicas=world_size if args.dist else 1))

    train_metrics = CompositeEvalMetric()
    val_metrics = CompositeEvalMetric()
    for child_metric in train_metrics_list:
        train_metrics.add(child_metric)
    for child_metric in val_metrics_list:
        val_metrics.add(child_metric)

    # epoch end callbacks
    epoch_end_callbacks = []
    if (rank is None) or (rank == 0):
        epoch_end_callbacks = [Checkpoint(model_prefix, config.CHECKPOINT_FREQUENT)]
    validation_monitor = ValidationMonitor(do_validation, val_loader, val_metrics,
                                           host_metric_name='RefAcc',
                                           label_index_in_batch=config.DATASET.LABEL_INDEX_IN_BATCH)

    # optimizer initial lr before
    for group in optimizer.param_groups:
        group.setdefault('initial_lr', group['lr'])

    # resume/auto-resume
    if rank is None or rank == 0:
        smart_resume(model, optimizer, validation_monitor, config, model_prefix, logger)
    if args.dist:
        begin_epoch = torch.tensor(config.TRAIN.BEGIN_EPOCH).cuda()
        distributed.broadcast(begin_epoch, src=0)
        config.TRAIN.BEGIN_EPOCH = begin_epoch.item()

    # batch end callbacks
    batch_size = len(config.GPUS.split(',')) * config.TRAIN.BATCH_IMAGES
    batch_end_callbacks = [Speedometer(batch_size, config.LOG_FREQUENT,
                                       batches_per_epoch=len(train_loader),
                                       epochs=config.TRAIN.END_EPOCH - config.TRAIN.BEGIN_EPOCH)]

    # setup lr step and lr scheduler
    if config.TRAIN.LR_SCHEDULE == 'plateau':
        print("Warning: not support resuming on plateau lr schedule!")
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                  mode='max',
                                                                  factor=config.TRAIN.LR_FACTOR,
                                                                  patience=1,
                                                                  verbose=True,
                                                                  threshold=1e-4,
                                                                  threshold_mode='rel',
                                                                  cooldown=2,
                                                                  min_lr=0,
                                                                  eps=1e-8)
    elif config.TRAIN.LR_SCHEDULE == 'triangle':
        lr_scheduler = WarmupLinearSchedule(optimizer,
                                            config.TRAIN.WARMUP_STEPS if config.TRAIN.WARMUP else 0,
                                            t_total=int(config.TRAIN.END_EPOCH * len(train_loader) / config.TRAIN.GRAD_ACCUMULATE_STEPS),
                                            last_epoch=int(config.TRAIN.BEGIN_EPOCH * len(train_loader) / config.TRAIN.GRAD_ACCUMULATE_STEPS)  - 1)
    elif config.TRAIN.LR_SCHEDULE == 'step':
        lr_iters = [int(epoch * len(train_loader) / config.TRAIN.GRAD_ACCUMULATE_STEPS) for epoch in config.TRAIN.LR_STEP]
        lr_scheduler = WarmupMultiStepLR(optimizer, milestones=lr_iters, gamma=config.TRAIN.LR_FACTOR,
                                         warmup_factor=config.TRAIN.WARMUP_FACTOR,
                                         warmup_iters=config.TRAIN.WARMUP_STEPS if config.TRAIN.WARMUP else 0,
                                         warmup_method=config.TRAIN.WARMUP_METHOD,
                                         last_epoch=int(config.TRAIN.BEGIN_EPOCH * len(train_loader) / config.TRAIN.GRAD_ACCUMULATE_STEPS)  - 1)
    else:
        raise ValueError("Not support lr schedule: {}.".format(config.TRAIN.LR_SCHEDULE))

    # broadcast parameter and optimizer state from rank 0 before training start
    if args.dist:
        for v in model.state_dict().values():
            distributed.broadcast(v, src=0)
        # for v in optimizer.state_dict().values():
        #     distributed.broadcast(v, src=0)
        best_epoch = torch.tensor(validation_monitor.best_epoch).cuda()
        best_val = torch.tensor(validation_monitor.best_val).cuda()
        distributed.broadcast(best_epoch, src=0)
        distributed.broadcast(best_val, src=0)
        validation_monitor.best_epoch = best_epoch.item()
        validation_monitor.best_val = best_val.item()

    # apex: amp fp16 mixed-precision training
    if config.TRAIN.FP16:
        # model.apply(bn_fp16_half_eval)
        model, optimizer = amp.initialize(model, optimizer,
                                          opt_level='O2',
                                          keep_batchnorm_fp32=False,
                                          loss_scale=config.TRAIN.FP16_LOSS_SCALE,
                                          min_loss_scale=128.0)
        if args.dist:
            model = Apex_DDP(model, delay_allreduce=True)

    train(model, optimizer, lr_scheduler, train_loader, train_sampler, train_metrics,
          config.TRAIN.BEGIN_EPOCH, config.TRAIN.END_EPOCH, logger,
          rank=rank, batch_end_callbacks=batch_end_callbacks, epoch_end_callbacks=epoch_end_callbacks,
          writer=writer, validation_monitor=validation_monitor, fp16=config.TRAIN.FP16,
          clip_grad_norm=config.TRAIN.CLIP_GRAD_NORM,
          gradient_accumulate_steps=config.TRAIN.GRAD_ACCUMULATE_STEPS)

    return rank, model
