from collections import namedtuple
import torch
from common.trainer import to_cuda


@torch.no_grad()
def do_validation(net, val_loader, metrics, label_index_in_batch):
    net.eval()
    metrics.reset()
    for nbatch, batch in enumerate(val_loader):
        batch = to_cuda(batch)
        label = batch[label_index_in_batch]
        datas = [batch[i] for i in range(len(batch)) if i != label_index_in_batch % len(batch)]

        outputs = net(*datas)
        outputs.update({'label': label})
        metrics.update(outputs)


@torch.no_grad()
def joint_validation(answer_net, rationale_net, answer_val_loader, rationale_val_loader, metrics, label_index_in_batch,
                     show_progress=False):
    answer_net.eval()
    rationale_net.eval()
    metrics.reset()

    def step(a_batch, r_batch):
        a_batch = to_cuda(a_batch)
        a_label = a_batch[label_index_in_batch]
        a_datas = [a_batch[i] for i in range(len(a_batch)) if i != label_index_in_batch % len(a_batch)]
        r_batch = to_cuda(r_batch)
        r_label = r_batch[label_index_in_batch]
        r_datas = [r_batch[i] for i in range(len(r_batch)) if i != label_index_in_batch % len(r_batch)]

        a_outputs = answer_net(*a_datas)
        r_outputs = rationale_net(*r_datas)
        outputs = {'answer_' + k: v for k, v in a_outputs.items()}
        outputs.update({'rationale_' + k: v for k, v in r_outputs.items()})
        outputs.update({'answer_label': a_label,
                        'rationale_label': r_label})
        metrics.update(outputs)

    if show_progress:
        from tqdm import tqdm
        for a_batch, r_batch in tqdm(zip(answer_val_loader, rationale_val_loader)):
            step(a_batch, r_batch)
    else:
        for a_batch, r_batch in zip(answer_val_loader, rationale_val_loader):
            step(a_batch, r_batch)
