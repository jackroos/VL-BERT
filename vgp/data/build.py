import torch.utils.data
import os

import sys
root_path = os.path.abspath(os.getcwd())
if root_path not in sys.path:
    sys.path.append(root_path)

from vgp.data.datasets import *
from vgp.data import samplers
from vgp.data.transforms.build import build_transforms
from vgp.data.collate_batch import BatchCollator
import pprint

DATASET_CATALOGS = {'vgp': VGPDataset}


def build_dataset(dataset_name, *args, **kwargs):
    assert dataset_name in DATASET_CATALOGS, "dataset not in catalogs"
    return DATASET_CATALOGS[dataset_name](*args, **kwargs)


def make_data_sampler(dataset, shuffle, distributed, num_replicas, rank):
    if distributed:
        return samplers.DistributedSampler(dataset, shuffle=shuffle, num_replicas=num_replicas, rank=rank)
    if shuffle:
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    return sampler


def make_batch_data_sampler(dataset, sampler, aspect_grouping, batch_size):
    if aspect_grouping:
        group_ids = dataset.group_ids
        batch_sampler = samplers.GroupedBatchSampler(
            sampler, group_ids, batch_size, drop_uneven=False
        )
    else:
        batch_sampler = torch.utils.data.sampler.BatchSampler(
            sampler, batch_size, drop_last=False
        )
    return batch_sampler


def make_dataloader(cfg, dataset=None, mode='train', distributed=False, num_replicas=None, rank=None,
                    expose_sampler=False):
    assert mode in ['train', 'val', 'test']
    if mode == 'train':
        ann_file = cfg.DATASET.TRAIN_ANNOTATION_FILE
        num_gpu = len(cfg.GPUS.split(','))
        batch_size = cfg.TRAIN.BATCH_IMAGES * num_gpu
        shuffle = cfg.TRAIN.SHUFFLE
        num_workers = cfg.NUM_WORKERS_PER_GPU * num_gpu
        captions_set = cfg.DATASET.TRAIN_CAPTIONS_SET
    elif mode == 'val':
        ann_file = cfg.DATASET.VAL_ANNOTATION_FILE
        num_gpu = len(cfg.GPUS.split(','))
        batch_size = cfg.VAL.BATCH_IMAGES * num_gpu
        shuffle = cfg.VAL.SHUFFLE
        num_workers = cfg.NUM_WORKERS_PER_GPU * num_gpu
        captions_set = cfg.DATASET.VAL_CAPTIONS_SET
    else:
        ann_file = cfg.DATASET.TEST_ANNOTATION_FILE
        num_gpu = len(cfg.GPUS.split(','))
        batch_size = cfg.TEST.BATCH_IMAGES * num_gpu
        shuffle = cfg.TEST.SHUFFLE
        num_workers = cfg.NUM_WORKERS_PER_GPU * num_gpu
        captions_set = cfg.DATASET.TEST_CAPTIONS_SET

    transform = build_transforms(cfg, mode)

    if dataset is None:
        kwargs = {}
        try:
            kwargs['with_lg'] = cfg.NETWORK.GNN.WITH_LG_LAYER
            kwargs['with_kg'] = cfg.NETWORK.GNN.WITH_KG
            kwargs['kg_path'] = cfg.DATASET.__getattribute__('{}_KG_PATH'.format(mode.upper()))
            kwargs['kg_word_embed'] = cfg.DATASET.__getattribute__('{}_KG_WORD_EMBED'.format(mode.upper()))
        except AttributeError:
            pass
        try:
            kwargs['kg_path'] = cfg.DATASET.__getattribute__('{}_KG_PATH'.format(mode.upper()))
            kwargs['fact_path'] = cfg.DATASET.__getattribute__('{}_KG_PATH'.format(mode.upper()))
        except AttributeError:
            pass
        try:
            kwargs['expression_file'] = cfg.DATASET.__getattribute__('{}_EXPRESSION_FILE'.format(mode.upper()))
        except AttributeError:
            pass

        try:
            kwargs['kg_vocab_file'] = cfg.NETWORK.KB_NODE_VOCAB
        except AttributeError:
            pass

        try:
            kwargs['caption_file'] = cfg.DATASET.__getattribute__('{}_CAPTION_FILE'.format(mode.upper()))
        except AttributeError:
            pass

        print('Dataset kwargs:')
        pprint.pprint(kwargs)
        dataset = build_dataset(dataset_name=cfg.DATASET.DATASET, captions_set=captions_set,
                                ann_file=ann_file, roi_set=cfg.DATASET.ROI_SET, image_set=cfg.DATASET.IMAGE_SET,
                                root_path=cfg.DATASET.ROOT_PATH, data_path=cfg.DATASET.DATASET_PATH,
                                transform=transform, test_mode=(mode == 'test'),  zip_mode=cfg.DATASET.ZIP_MODE,
                                cache_mode=cfg.DATASET.CACHE_MODE, ignore_db_cache=cfg.DATASET.IGNORE_DB_CACHE,
                                pretrained_model_name=cfg.NETWORK.BERT_MODEL_NAME,
                                add_image_as_a_box=cfg.DATASET.ADD_IMAGE_AS_A_BOX,
                                **kwargs)

    sampler = make_data_sampler(dataset, shuffle, distributed, num_replicas, rank)
    batch_sampler = make_batch_data_sampler(dataset, sampler, False, batch_size)
    collator = BatchCollator(dataset=dataset, append_ind=cfg.DATASET.APPEND_INDEX)

    dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                             batch_sampler=batch_sampler,
                                             num_workers=num_workers,
                                             pin_memory=False,
                                             collate_fn=collator)
    if expose_sampler:
        return dataloader, sampler

    return dataloader


def test_dataloader():
    from vgp.function.config import config, update_config
    cfg_path = os.path.join(root_path, 'cfgs', 'vgp', 'base_4x16G_fp32.yaml')
    update_config(cfg_path)
    dataloader = make_dataloader(config, dataset=None, mode='train')
    for batch in dataloader:
        print(len(batch))


if __name__ == '__main__':
    test_dataloader()
