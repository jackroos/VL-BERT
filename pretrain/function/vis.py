import os
import pprint
import shutil
import inspect
import random
import math

from tqdm import trange
import numpy as np
import torch
import torch.nn
import torch.distributed as distributed
from torch.nn.parallel import DistributedDataParallel as DDP

from common.utils.load import smart_partial_load_model_state_dict
from common.trainer import to_cuda
from common.utils.multi_task_dataloader import MultiTaskDataLoader
from pretrain.data.build import make_dataloader, make_dataloaders
from pretrain.modules import *
from common.utils.create_logger import makedirsExist


def vis_net(args, config, save_dir):
    pprint.pprint(config)

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
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

        if isinstance(config.DATASET, list):
            val_loaders = make_dataloaders(config,
                                           mode='val',
                                           distributed=True,
                                           num_replicas=world_size,
                                           rank=rank)
            val_loader = MultiTaskDataLoader(val_loaders)
        else:
            val_loader = make_dataloader(config,
                                         mode='val',
                                         distributed=True,
                                         num_replicas=world_size,
                                         rank=rank)
    else:
        model = eval(config.MODULE)(config)
        num_gpus = len(config.GPUS.split(','))
        rank = None
        # model
        if num_gpus > 1:
            model = torch.nn.DataParallel(model, device_ids=[int(d) for d in config.GPUS.split(',')]).cuda()
        else:
            torch.cuda.set_device(int(config.GPUS))
            model.cuda()

        # loader
        if isinstance(config.DATASET, list):
            val_loaders = make_dataloaders(config, mode='val', distributed=False)
            val_loader = MultiTaskDataLoader(val_loaders)
        else:
            val_loader = make_dataloader(config, mode='val', distributed=False)

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

    # broadcast parameter and optimizer state from rank 0 before training start
    if args.dist:
        for v in model.state_dict().values():
            distributed.broadcast(v, src=0)

    vis(model, val_loader, save_dir, rank=rank, world_size=world_size if args.dist else 1)

    return rank, model


def vis(model, loader, save_dir, rank=None, world_size=1):
    attention_dir = os.path.join(save_dir, 'attention_probs')
    hidden_dir = os.path.join(save_dir, 'hidden_states')
    cos_dir = os.path.join(save_dir, 'cos_similarity')
    # if not os.path.exists(hidden_dir):
    #     makedirsExist(hidden_dir)
    # if not os.path.exists(cos_dir):
    #     makedirsExist(cos_dir)
    if not os.path.exists(attention_dir):
        makedirsExist(attention_dir)
    # offset = 0
    # if rank is not None:
    #     num_samples = int(math.ceil(len(loader.dataset) * 1.0 / world_size))
    #     offset = num_samples * rank
    # index = offset
    model.eval()
    for i, data in zip(trange(len(loader)), loader):
    # for i, data in enumerate(loader):
        data = to_cuda(data)
        output = model(*data)
        for _i, (attention_probs, hidden_states) in enumerate(zip(output['attention_probs'], output['hidden_states'])):
            index = int(data[2][_i][-1])
            if hasattr(loader.dataset, 'ids'):
                image_id = loader.dataset.ids[index]
            else:
                image_id = loader.dataset.database[index]['image'].split('/')[1].split('.')[0]
            attention_probs_arr = attention_probs.detach().cpu().numpy()
            hidden_states_arr = hidden_states.detach().cpu().numpy()
            cos_similarity_arr = (hidden_states @ hidden_states.transpose(1, 2)).detach().cpu().numpy()
            np.save(os.path.join(attention_dir, '{}.npy'.format(image_id)), attention_probs_arr)
            # np.save(os.path.join(hidden_dir, '{}.npy'.format(image_id)), hidden_states_arr)
            # np.save(os.path.join(cos_dir, '{}.npy'.format(image_id)), cos_similarity_arr)
            # index = (index + 1) % len(loader.dataset)



