from collections import namedtuple
import torch
from common.trainer import to_cuda


@torch.no_grad()
def do_validation(net, val_loader, metrics, label_index_in_batch):
    net.eval()
    metrics.reset()
    for nbatch, batch in enumerate(val_loader):
        batch = to_cuda(batch)
        outputs, _ = net(*batch)
        metrics.update(outputs)

