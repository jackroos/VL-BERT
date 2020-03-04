import _init_paths
import os
import argparse
import torch
import subprocess

from pretrain.function.config import config, update_config
from pretrain.function.vis import vis_net


def parse_args():
    parser = argparse.ArgumentParser('Visualize Attention Maps')
    parser.add_argument('--cfg', type=str, help='path to config file')
    parser.add_argument('--dist', help='whether to use distributed training', default=False, action='store_true')
    parser.add_argument('--slurm', help='whether this is a slurm job', default=False, action='store_true')
    parser.add_argument('--save-dir', help='directory to save attention maps', type=str, default='./attention_maps')

    args = parser.parse_args()

    if args.cfg is not None:
        update_config(args.cfg)

    if args.slurm:
        proc_id = int(os.environ['SLURM_PROCID'])
        ntasks = int(os.environ['SLURM_NTASKS'])
        node_list = os.environ['SLURM_NODELIST']
        num_gpus = torch.cuda.device_count()
        addr = subprocess.getoutput(
            'scontrol show hostname {} | head -n1'.format(node_list))
        os.environ['MASTER_PORT'] = str(29500)
        os.environ['MASTER_ADDR'] = addr
        os.environ['WORLD_SIZE'] = str(ntasks)
        os.environ['RANK'] = str(proc_id)
        os.environ['LOCAL_RANK'] = str(proc_id % num_gpus)

    return args, config


def main():
    args, config = parse_args()
    rank, model = vis_net(args, config, args.save_dir)


if __name__ == '__main__':
    main()