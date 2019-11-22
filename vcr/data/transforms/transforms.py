import random

import numpy as np

import torch
import torchvision
from torchvision.transforms import functional as F


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, boxes, masks, im_info):
        for t in self.transforms:
            image, boxes, masks, im_info = t(image, boxes, masks, im_info)
        return image, boxes, masks, im_info

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class Resize(object):
    def __init__(self, min_size, max_size):
        self.min_size = min_size
        self.max_size = max_size

    # modified from torchvision to add support for max size
    def get_size(self, image_size):
        w, h = image_size
        size = self.min_size
        max_size = self.max_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(max_size * min_original_size / max_original_size)

        if (w <= h and w == size) or (h <= w and h == size):
            return (w, h)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (ow, oh)

    def __call__(self, image, boxes, masks, im_info):
        origin_size = image.size
        size = self.get_size(origin_size)
        image = F.resize(image, (size[1], size[0]))

        ratios = [size[0] * 1.0 / origin_size[0], size[1] * 1.0 / origin_size[1]]
        boxes[:, [0, 2]] *= ratios[0]
        boxes[:, [1, 3]] *= ratios[1]
        im_info[0], im_info[1] = image.size
        im_info[2], im_info[3] = ratios
        return image, boxes, masks, im_info


class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, boxes, masks, im_info):
        if random.random() < self.prob:
            w, h = image.size
            image = F.hflip(image)
            boxes[:, [0, 2]] = w - 1 - boxes[:, [2, 0]]
            masks = torch.as_tensor(masks.numpy()[:, :, ::-1].tolist())
        return image, boxes, masks, im_info


class ToTensor(object):
    def __call__(self, image, boxes, masks, im_info):
        return F.to_tensor(image), boxes, masks, im_info


class Normalize(object):
    def __init__(self, mean, std, to_bgr255=True):
        self.mean = mean
        self.std = std
        self.to_bgr255 = to_bgr255

    def __call__(self, image, boxes, masks, im_info):
        if self.to_bgr255:
            image = image[[2, 1, 0]] * 255
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, boxes, masks, im_info


class FixPadding(object):
    def __init__(self, min_size, max_size, pad=0):
        self.min_size = min_size
        self.max_size = max_size
        self.pad = pad

    def __call__(self, image, boxes, masks, im_info):

        # padding to fixed size for determinacy
        c, h, w = image.shape
        if h <= w:
            h1 = self.min_size
            w1 = self.max_size
        else:
            h1 = self.max_size
            w1 = self.min_size
        padded_image = image.new_zeros((c, h1, w1)).fill_(self.pad)
        padded_image[:, :h, :w] = image
        image = padded_image

        return image, boxes, masks, im_info
