import sys
import os
import argparse

root_path = os.path.abspath(os.getcwd())
if root_path not in sys.path:
    sys.path.append(root_path)

from easydict import EasyDict as edict
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.neighbors import NearestNeighbors

from common.backbone.resnet.resnet import resnet101
from common.utils.clip_pad import *
from vgp.data.transforms.build import build_transforms


def parseArguments():
    # Create argument parser
    parser = argparse.ArgumentParser()

    # Positional mandatory arguments
    parser.add_argument("captionspath", help="Path to folder that contains all image ids to use", type=str)
    parser.add_argument("imgpath", help="Path to folder containing all the flickr30k images", type=str)
    parser.add_argument("modelpath", help="Path to pretained model parameters", type=str)

    # Optional arguments
    parser.add_argument("-b", "--bs", help="Batch size", type=int, default=64)
    parser.add_argument("-n", "--nn", help="Number of nearest neighbors to use", type=int, default=100)
    parser.add_argument("-ex", "--reextract", help="Whether to ignore saved image features", action="store_true")

    # Parse arguments
    args = parser.parse_args()

    return args


class BatchCollator(object):
    def __init__(self, dataset):
        self.dataset = dataset

    def __call__(self, batch):
        
        max_shape = tuple(max(s) for s in zip(*[data.shape for data in batch]))
        batch = torch.cat([clip_pad_images(image, max_shape, pad=0).unsqueeze(0) for image in batch])
        
        return batch




class Flickr30k_imgDataset(Dataset):
    def __init__(self, captions_path, img_path, transform):
        super(Flickr30k_imgDataset, self).__init__()

        self.ids = [file[:-4] for file in os.listdir(captions_path) if file.endswith(".txt")]
        self.img_path = img_path
        self.transform = transform

    def __getitem__(self, index):
        img_id = self.ids[index]
        image = Image.open(os.path.join(self.img_path, img_id + ".jpg"))
        if self.transform is not None:
            image, _, _, _ = self.transform(image, None, None, None)
        return image

    def __len__(self):
        return len(self.ids)


def create_cfg():
    cfg = edict()
    cfg.SCALES = (600, 1000)
    cfg.TEST = edict()
    cfg.TEST.FLIP_PROB = 0
    cfg.NETWORK = edict()
    cfg.NETWORK.PIXEL_MEANS = (102.9801, 115.9465, 122.7717)
    cfg.NETWORK.PIXEL_STDS = (1.0, 1.0, 1.0)
    cfg.DATASET = edict()
    cfg.DATASET.FIX_PADDING = True
    return cfg


def get_img_features(captions_path, img_path, model_path, batch_size):
    cfg = create_cfg()
    transform = build_transforms(cfg, mode="test")
    dataset = Flickr30k_imgDataset(captions_path, img_path, transform)
    batch_collator = BatchCollator(dataset)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, collate_fn=batch_collator, pin_memory=False,
                            drop_last=False, shuffle=False)
    model = resnet101(pretrained=True, pretrained_model_path=model_path, expose_stages=[5],
                      stride_in_1x1=True).eval().cuda()
    avg_pooler = nn.AdaptiveAvgPool2d((1, 1))
    img_features = []
    for batch in dataloader:
        batch = batch.cuda(non_blocking=True)
        features = avg_pooler(model(batch)['body5']).detach().float().cpu().numpy()
        img_features.append(features)
    img_features = np.concatenate(img_features, axis=0).squeeze()
    np.save(os.path.join(captions_path, "extracted_img_features"), img_features)
    return img_features


def main(captions_path, img_path, model_path, batch_size, n_neighbors, use_saved=True):
    list_ids = [file[:-4] for file in os.listdir(captions_path)  if file.endswith(".txt")]
    if os.path.exists(os.path.join(captions_path, "extracted_img_features.npy")) and use_saved:
        print("loading saved features")
        img_features = np.load(os.path.join(captions_path, "extracted_img_features.npy"))
    else:
        print("image features not found, start extracting them")
        img_features = get_img_features(captions_path, img_path, model_path, batch_size)
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(img_features)
    nearest_imgs = nbrs.kneighbors(img_features, return_distance=False)
    result = np.concatenate((np.array(list_ids).reshape((-1, 1)), nearest_imgs), axis=1)
    column_names = ["img_id"] + [str(k) for k in range(1, n_neighbors + 1)]
    nearest_imgs_df = pd.DataFrame(result, columns=column_names)
    nearest_imgs_df.to_csv(os.path.join(captions_path, "similarities.csv"))


if __name__ == "__main__":
    args = parseArguments()
    root_path = os.getcwd()
    data_folder = os.path.join(root_path, args.captionspath)
    img_folder = os.path.join(root_path, args.imgpath)
    trained_model = os.path.join(root_path, args.modelpath)
    bs = args.bs
    n_neighbors = args.nn
    use_saved = not args.reextract
    main(data_folder, img_folder, trained_model, bs, n_neighbors, use_saved)
