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

        if 'image' in self.data_names:
            if batch[0][self.data_names.index('image')] is not None:
                max_shape = tuple(max(s) for s in zip(*[data[self.data_names.index('image')].shape for data in batch]))
                image_none = False
            else:
                image_none = True
        if 'boxes' in self.data_names:
            max_boxes = max([data[self.data_names.index('boxes')].shape[0] for data in batch])
        if 'text' in self.data_names:
            max_text_length = max([len(data[self.data_names.index('text')]) for data in batch])

        for i, ibatch in enumerate(batch):
            out = {}

            if 'image' in self.data_names:
                if image_none:
                    out['image'] = None
                else:
                    image = ibatch[self.data_names.index('image')]
                    out['image'] = clip_pad_images(image, max_shape, pad=0)

            if 'boxes' in self.data_names:
                boxes = ibatch[self.data_names.index('boxes')]
                out['boxes'] = clip_pad_boxes(boxes, max_boxes, pad=-2)

            if 'text' in self.data_names:
                text = ibatch[self.data_names.index('text')]
                out['text'] = clip_pad_1d(text, max_text_length, pad=0)

            if 'mlm_labels' in self.data_names:
                mlm_labels = ibatch[self.data_names.index('mlm_labels')]
                out['mlm_labels'] = clip_pad_1d(mlm_labels, max_text_length, pad=-1)

            if 'mvrc_ops' in self.data_names:
                mvrc_ops = ibatch[self.data_names.index('mvrc_ops')]
                out['mvrc_ops'] = clip_pad_1d(mvrc_ops, max_boxes, pad=0)

            if 'mvrc_labels' in self.data_names:
                mvrc_labels = ibatch[self.data_names.index('mvrc_labels')]
                out['mvrc_labels'] = clip_pad_boxes(mvrc_labels, max_boxes, pad=0)

            other_names = [data_name for data_name in self.data_names if data_name not in out]
            for name in other_names:
                out[name] = torch.as_tensor(ibatch[self.data_names.index(name)])

            batch[i] = tuple(out[data_name] for data_name in self.data_names)
            if self.append_ind:
                batch[i] += (torch.tensor(i, dtype=torch.int64),)

        out_tuple = ()
        for items in zip(*batch):
            if items[0] is None:
                out_tuple += (None,)
            else:
                out_tuple += (torch.stack(tuple(items), dim=0), )

        return out_tuple

