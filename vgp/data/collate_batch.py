import torch

from common.utils.clip_pad import *


class BatchCollator(object):
    def __init__(self, dataset, append_ind=False):
        self.dataset = dataset
        self.test_mode = self.dataset.test_mode
        self.data_names = self.dataset.data_names
        self.append_ind = append_ind

    def __call__(self, batch):
        if not isinstance(batch, list):
            batch = list(batch)

        max_shape = tuple(max(s) for s in zip(*[data[self.data_names.index('image')].shape for data in batch]))
        max_boxes = max([data[self.data_names.index('boxes')].shape[0] for data in batch])
        max_caption1_length = max([len(data[self.data_names.index('caption1')]) for data in batch])
        max_caption2_length = max([len(data[self.data_names.index('caption2')]) for data in batch])

        for i, ibatch in enumerate(batch):
            out = {}
            image = ibatch[self.data_names.index('image')]
            out['image'] = clip_pad_images(image, max_shape, pad=0)

            boxes = ibatch[self.data_names.index('boxes')]
            out['boxes'] = clip_pad_boxes(boxes, max_boxes, pad=-1)

            caption1 = ibatch[self.data_names.index('caption1')]
            caption2 = ibatch[self.data_names.index('caption2')]
            out['caption1'] = clip_pad_2d(caption1, (max_caption1_length, len(caption1[0])), pad=0)
            out['caption2'] = clip_pad_2d(caption2, (max_caption2_length, len(caption2[0])), pad=0)

            out['im_info'] = ibatch[self.data_names.index('im_info')]
            if 'label' in self.data_names:
                out['label'] = ibatch[self.data_names.index('label')]
            if 'first_correct' in self.data_names:
                out['first_correct'] = ibatch[self.data_names.index('first_correct')]

            other_names = [data_name for data_name in self.data_names if data_name not in out]
            for name in other_names:
                out[name] = torch.as_tensor(ibatch[self.data_names.index(name)])

            batch[i] = tuple(out[data_name] for data_name in self.data_names)
            if self.append_ind:
                batch[i] += (torch.tensor(i, dtype=torch.int64),)

        out_tuple = ()
        for items in zip(*batch):
            if isinstance(items[0], torch.Tensor):
                out_tuple += (torch.stack(tuple(items), dim=0), )
            else:
                out_tuple += (list(items), )

        return out_tuple

