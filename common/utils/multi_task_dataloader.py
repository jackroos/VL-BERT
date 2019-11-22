from functools import reduce
import operator
from typing import List
from torch.utils.data import DataLoader
import sys

INT_MAX = sys.maxsize


def prod(iterable):
    if len(list(iterable)) > 0:
        return reduce(operator.mul, iterable)
    else:
        return 1


class MultiTaskDataLoader(object):
    """
    Multi-task DataLoader, the first dataloader is master dataloader
    """
    def __init__(self,
                 loaders: List[DataLoader]):
        assert len(loaders) > 1, "Less than 2 loader!"
        self.loaders = loaders
        self.iters = [iter(loader) for loader in loaders]
        self.lens = [len(loader) for loader in loaders]
        self.global_idx_in_cycle = 0

    def __iter__(self):
        if self.global_idx_in_cycle > 0:
            self.iters[0] = iter(self.loaders[0])
        return self

    def __next__(self):
        output_tuple = (*next(self.iters[0]), )
        for k, (loader, _iter) in enumerate(zip(self.loaders[1:], self.iters[1:])):
            if hasattr(loader.batch_sampler.sampler, 'set_epoch'):
                loader.batch_sampler.sampler.set_epoch(int(self.global_idx_in_cycle / self.lens[k+1]))
            try:
                output_tuple += (*next(_iter), )
            except StopIteration:
                _iter = iter(loader)
                self.iters[k+1] = _iter
                output_tuple += (*next(_iter), )

        if self.global_idx_in_cycle < INT_MAX - 1:
            self.global_idx_in_cycle += 1
        else:
            self.global_idx_in_cycle = 0

        return output_tuple

    def __len__(self):
        return self.lens[0]



