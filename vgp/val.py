import _init_paths
import os
import argparse
from copy import deepcopy

import jsonlines
from tqdm import trange
import torch
import numpy as np

from common.utils.load import smart_load_model_state_dict
from common.trainer import to_cuda
from common.metrics.composite_eval_metric import CompositeEvalMetric
from common.metrics.vcr_metrics import JointAccuracy
from vcr.data.build import make_dataloader
from vcr.function.config import config, update_config
from vcr.modules import *
from vcr.function.val import joint_validation

try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as Apex_DDP
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex if you want to use fp16.")


def parse_args():
    parser = argparse.ArgumentParser('Do Validation of Cognition Network')
    parser.add_argument('--a-cfg', type=str, help='path to answer net config yaml')
    parser.add_argument('--r-cfg', type=str, help='path to rationale net config yaml')
    parser.add_argument('--a-ckpt', type=str, help='path to checkpoint of answer net')
    parser.add_argument('--r-ckpt', type=str, help='path to checkpoint of rationale net')
    parser.add_argument('--a-bs', type=int)
    parser.add_argument('--r-bs', type=int)
    parser.add_argument('--gpus', type=int, nargs='+', default=[0])
    parser.add_argument('--result-path', type=str, default='./vcr_val_results')
    parser.add_argument('--result-name', type=str, default='vl-bert')
    parser.add_argument('--use-cache', default=False, action='store_true')
    parser.add_argument('--annot', type=str, default='./data/vcr/val.jsonl')
    parser.add_argument('--cudnn-off', default=False, action='store_true')
    parser.add_argument('--fp16', default=False, action='store_true')

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
        a_config.VAL.BATCH_IMAGES = args.a_bs
    if args.r_bs is not None:
        r_config.VAL.BATCH_IMAGES = args.r_bs

    return args, a_config, r_config


@torch.no_grad()
def main():
    args, a_config, r_config = parse_args()

    if args.cudnn_off:
        torch.backends.cudnn.enabled = False

    with jsonlines.open(args.annot) as reader:
        gts = [(obj['answer_label'], obj['rationale_label']) for obj in reader]
    a_gt = np.array([gt[0] for gt in gts], dtype=np.int64)
    r_gt = np.array([gt[1] for gt in gts], dtype=np.int64)

    # cache
    a_cache_fn = os.path.join(args.result_path, '{}_a_pred.npy'.format(args.result_name))
    r_cache_fn = os.path.join(args.result_path, '{}_r_pred.npy'.format(args.result_name))
    a_pred = r_pred = None
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)
    if args.use_cache:
        if os.path.exists(a_cache_fn):
            print("Load cached predictions from {}...".format(a_cache_fn))
            a_pred = np.load(a_cache_fn)
        if os.path.exists(r_cache_fn):
            print("Load cached predictions from {}...".format(r_cache_fn))
            r_pred = np.load(r_cache_fn)
    else:
        if a_config is not None and args.a_ckpt is not None:

            print("Build model and dataloader for Q->A...")

            # get model
            device_ids = args.gpus
            # os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(k) for k in args.gpus])
            a_config.GPUS = ','.join([str(k) for k in args.gpus])
            answer_model = eval(a_config.MODULE)(a_config)
            if len(device_ids) > 1:
                answer_model = torch.nn.DataParallel(answer_model, device_ids=device_ids).cuda()
            else:
                torch.cuda.set_device(device_ids[0])
                answer_model = answer_model.cuda()

            if args.fp16:
                [answer_model] = amp.initialize([answer_model],
                                                opt_level='O2',
                                                keep_batchnorm_fp32=False)

            a_ckpt = torch.load(args.a_ckpt, map_location=lambda storage, loc: storage)
            smart_load_model_state_dict(answer_model, a_ckpt['state_dict'])
            answer_model.eval()

            # get data loader
            a_config.DATASET.TASK = 'Q2A'
            a_config.VAL.SHUFFLE = False
            answer_loader = make_dataloader(a_config, mode='val', distributed=False)
            label_index_in_batch = a_config.DATASET.LABEL_INDEX_IN_BATCH

            print("Inference Q->A...")

            # inference
            n_batch = len(answer_loader)
            a_pred = np.zeros((len(gts), 4), dtype=np.float)
            i_sample = 0
            for nbatch, a_batch in zip(trange(len(answer_loader)), answer_loader):
            # for a_batch in answer_loader:
                a_batch = to_cuda(a_batch)
                a_batch = [a_batch[i] for i in range(len(a_batch)) if i != label_index_in_batch % len(a_batch)]
                a_out = answer_model(*a_batch)
                a_batch_pred = a_out['label_logits']
                batch_size = a_batch_pred.shape[0]
                if a_batch_pred.dim() == 2:
                    a_pred[i_sample:(i_sample + batch_size)] = a_batch_pred.detach().cpu().numpy().astype(np.float,
                                                                                                          copy=False)
                elif a_batch_pred.dim() == 1:
                    assert a_batch_pred.shape[0] % 4 == 0
                    a_batch_pred = a_batch_pred.view((-1, 4))
                    a_pred[int(i_sample / 4):int((i_sample + batch_size) / 4)] \
                        = a_batch_pred.float().detach().cpu().numpy().astype(np.float, copy=False)
                else:
                    raise ValueError("Invalid")
                i_sample += batch_size
                # print("inference {}/{}".format(i_sample, len(answer_loader.dataset)))
            np.save(a_cache_fn, a_pred)

        if r_config is not None and args.r_ckpt is not None:

            print("Build model and dataloader for QA->R...")

            # get model
            device_ids = args.gpus
            # os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(k) for k in args.gpus])
            r_config.GPUS = ','.join([str(k) for k in args.gpus])
            rationale_model = eval(r_config.MODULE)(r_config)
            if len(device_ids) > 1:
                rationale_model = torch.nn.DataParallel(rationale_model, device_ids=device_ids).cuda()
            else:
                torch.cuda.set_device(device_ids[0])
                rationale_model = rationale_model.cuda()

            if args.fp16:
                [rationale_model] = amp.initialize([rationale_model],
                                                   opt_level='O2',
                                                   keep_batchnorm_fp32=False)

            r_ckpt = torch.load(args.r_ckpt, map_location=lambda storage, loc: storage)
            smart_load_model_state_dict(rationale_model, r_ckpt['state_dict'])
            rationale_model.eval()

            # get data loader
            r_config.DATASET.TASK = 'QA2R'
            r_config.VAL.SHUFFLE = False
            rationale_loader = make_dataloader(r_config, mode='val', distributed=False)
            label_index_in_batch = r_config.DATASET.LABEL_INDEX_IN_BATCH

            print("Inference QA->R...")

            # inference
            n_batch = len(rationale_loader)
            r_pred = np.zeros((len(rationale_loader.dataset), 4), dtype=np.float)
            i_sample = 0
            for nbatch, r_batch in zip(trange(len(rationale_loader)), rationale_loader):
            # for r_batch in rationale_loader:
                r_batch = to_cuda(r_batch)
                r_batch = [r_batch[i] for i in range(len(r_batch)) if i != label_index_in_batch % len(r_batch)]
                r_out = rationale_model(*r_batch)
                r_batch_pred = r_out['label_logits']
                batch_size = r_batch_pred.shape[0]
                r_pred[i_sample:(i_sample + batch_size)] =\
                    r_batch_pred.float().detach().cpu().numpy().astype(np.float, copy=False)
                i_sample += batch_size
                # print("inference {}/{}".format(i_sample, len(rationale_loader.dataset)))
            np.save(r_cache_fn, r_pred)

    # evaluate
    print("Evaluate...")
    if a_pred is not None:
        acc_a = (a_pred.argmax(1) == a_gt).sum() * 1.0 / a_gt.size
        print("Q->A\t{:.1f}".format(acc_a * 100.0))
    if r_pred is not None:
        acc_r = (r_pred.argmax(1) == r_gt).sum() * 1.0 / r_gt.size
        print("QA->R\t{:.1f}".format(acc_r * 100.0))
    if a_pred is not None and r_pred is not None:
        acc_joint = ((a_pred.argmax(1) == a_gt) * (r_pred.argmax(1) == r_gt)).sum() * 1.0 / a_gt.size
        print("Q->AR\t{:.1f}".format(acc_joint * 100.0))


if __name__ == '__main__':
    main()


