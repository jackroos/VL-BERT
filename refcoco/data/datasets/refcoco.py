import os
import json
import _pickle as cPickle
from PIL import Image
import base64
import numpy as np
import time
import logging

import torch
from torch.utils.data import Dataset
from external.pytorch_pretrained_bert import BertTokenizer

from common.utils.zipreader import ZipReader
from common.utils.create_logger import makedirsExist
from common.utils.bbox import bbox_iou_py_vectorized

from pycocotools.coco import COCO
from .refer.refer import REFER


class RefCOCO(Dataset):
    def __init__(self, image_set, root_path, data_path, boxes='gt', proposal_source='official',
                 transform=None, test_mode=False,
                 zip_mode=False, cache_mode=False, cache_db=False, ignore_db_cache=True,
                 tokenizer=None, pretrained_model_name=None,
                 add_image_as_a_box=False, mask_size=(14, 14),
                 aspect_grouping=False, **kwargs):
        """
        RefCOCO+ Dataset

        :param image_set: image folder name
        :param root_path: root path to cache database loaded from annotation file
        :param data_path: path to dataset
        :param boxes: boxes to use, 'gt' or 'proposal'
        :param transform: transform
        :param test_mode: test mode means no labels available
        :param zip_mode: reading images and metadata in zip archive
        :param cache_mode: cache whole dataset to RAM first, then __getitem__ read them from RAM
        :param ignore_db_cache: ignore previous cached database, reload it from annotation file
        :param tokenizer: default is BertTokenizer from pytorch_pretrained_bert
        :param add_image_as_a_box: add whole image as a box
        :param mask_size: size of instance mask of each object
        :param aspect_grouping: whether to group images via their aspect
        :param kwargs:
        """
        super(RefCOCO, self).__init__()

        assert not cache_mode, 'currently not support cache mode!'

        categories = ['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
                      'boat',
                      'trafficlight', 'firehydrant', 'stopsign', 'parkingmeter', 'bench', 'bird', 'cat', 'dog', 'horse',
                      'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                      'suitcase', 'frisbee', 'skis', 'snowboard', 'sportsball', 'kite', 'baseballbat', 'baseballglove',
                      'skateboard', 'surfboard', 'tennisracket', 'bottle', 'wineglass', 'cup', 'fork', 'knife', 'spoon',
                      'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hotdog', 'pizza', 'donut',
                      'cake', 'chair', 'couch', 'pottedplant', 'bed', 'diningtable', 'toilet', 'tv', 'laptop', 'mouse',
                      'remote', 'keyboard', 'cellphone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
                      'clock', 'vase', 'scissors', 'teddybear', 'hairdrier', 'toothbrush']

        coco_annot_files = {
            "train2014": "annotations/instances_train2014.json",
            "val2014": "annotations/instances_val2014.json",
            "test2015": "annotations/image_info_test2015.json",
        }
        proposal_dets = 'refcoco+/proposal/res101_coco_minus_refer_notime_dets.json'
        proposal_masks = 'refcoco+/proposal/res101_coco_minus_refer_notime_masks.json'
        self.vg_proposal = ("vgbua_res101_precomputed", "trainval2014_resnet101_faster_rcnn_genome")
        self.proposal_source = proposal_source
        self.boxes = boxes
        self.test_mode = test_mode
        self.category_to_idx = {c: i for i, c in enumerate(categories)}
        self.data_path = data_path
        self.root_path = root_path
        self.transform = transform
        self.image_sets = [iset.strip() for iset in image_set.split('+')]
        self.coco = COCO(annotation_file=os.path.join(data_path, coco_annot_files['train2014']))
        self.refer = REFER(data_path, dataset='refcoco+', splitBy='unc')
        self.refer_ids = []
        for iset in self.image_sets:
            self.refer_ids.extend(self.refer.getRefIds(split=iset))
        self.refs = self.refer.loadRefs(ref_ids=self.refer_ids)
        if 'proposal' in boxes:
            with open(os.path.join(data_path, proposal_dets), 'r') as f:
                proposal_list = json.load(f)
            self.proposals = {}
            for proposal in proposal_list:
                image_id = proposal['image_id']
                if image_id in self.proposals:
                    self.proposals[image_id].append(proposal['box'])
                else:
                    self.proposals[image_id] = [proposal['box']]
        self.zip_mode = zip_mode
        self.cache_mode = cache_mode
        self.cache_db = cache_db
        self.ignore_db_cache = ignore_db_cache
        self.aspect_grouping = aspect_grouping
        self.cache_dir = os.path.join(root_path, 'cache')
        self.add_image_as_a_box = add_image_as_a_box
        self.mask_size = mask_size
        if not os.path.exists(self.cache_dir):
            makedirsExist(self.cache_dir)
        self.tokenizer = tokenizer if tokenizer is not None \
            else BertTokenizer.from_pretrained(
            'bert-base-uncased' if pretrained_model_name is None else pretrained_model_name,
            cache_dir=self.cache_dir)

        if zip_mode:
            self.zipreader = ZipReader()

        self.database = self.load_annotations()
        if self.aspect_grouping:
            self.group_ids = self.group_aspect(self.database)

    @property
    def data_names(self):
        if self.test_mode:
            return ['image', 'boxes', 'im_info', 'expression']
        else:
            return ['image', 'boxes', 'im_info', 'expression', 'label']

    def __getitem__(self, index):
        idb = self.database[index]

        # image related
        img_id = idb['image_id']
        image = self._load_image(idb['image_fn'])
        im_info = torch.as_tensor([idb['width'], idb['height'], 1.0, 1.0])
        if not self.test_mode:
            gt_box = torch.as_tensor(idb['gt_box'])
        flipped = False
        if self.boxes == 'gt':
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)
            boxes = []
            for ann in anns:
                x_, y_, w_, h_ = ann['bbox']
                boxes.append([x_, y_, x_ + w_, y_ + h_])
            boxes = torch.as_tensor(boxes)
        elif self.boxes == 'proposal':
            if self.proposal_source == 'official':
                boxes = torch.as_tensor(self.proposals[img_id])
                boxes[:, [2, 3]] += boxes[:, [0, 1]]
            elif self.proposal_source == 'vg':
                box_file = os.path.join(self.data_path, self.vg_proposal[0], '{0}.zip@/{0}'.format(self.vg_proposal[1]))
                boxes_fn = os.path.join(box_file, '{}.json'.format(idb['image_id']))
                boxes_data = self._load_json(boxes_fn)
                boxes = torch.as_tensor(
                    np.frombuffer(self.b64_decode(boxes_data['boxes']), dtype=np.float32).reshape(
                        (boxes_data['num_boxes'], -1))
                )
            else:
                raise NotImplemented
        elif self.boxes == 'proposal+gt' or self.boxes == 'gt+proposal':
            if self.proposal_source == 'official':
                boxes = torch.as_tensor(self.proposals[img_id])
                boxes[:, [2, 3]] += boxes[:, [0, 1]]
            elif self.proposal_source == 'vg':
                box_file = os.path.join(self.data_path, self.vg_proposal[0], '{0}.zip@/{0}'.format(self.vg_proposal[1]))
                boxes_fn = os.path.join(box_file, '{}.json'.format(idb['image_id']))
                boxes_data = self._load_json(boxes_fn)
                boxes = torch.as_tensor(
                    np.frombuffer(self.b64_decode(boxes_data['boxes']), dtype=np.float32).reshape(
                        (boxes_data['num_boxes'], -1))
                )
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)
            gt_boxes = []
            for ann in anns:
                x_, y_, w_, h_ = ann['bbox']
                gt_boxes.append([x_, y_, x_ + w_, y_ + h_])
            gt_boxes = torch.as_tensor(gt_boxes)
            boxes = torch.cat((boxes, gt_boxes), 0)
        else:
            raise NotImplemented

        if self.add_image_as_a_box:
            w0, h0 = im_info[0], im_info[1]
            image_box = torch.as_tensor([[0.0, 0.0, w0 - 1, h0 - 1]])
            boxes = torch.cat((image_box, boxes), dim=0)

        if self.transform is not None:
            if not self.test_mode:
                boxes = torch.cat((gt_box[None], boxes), 0)
            image, boxes, _, im_info, flipped = self.transform(image, boxes, None, im_info, flipped)
            if not self.test_mode:
                gt_box = boxes[0]
                boxes = boxes[1:]

        # clamp boxes
        w = im_info[0].item()
        h = im_info[1].item()
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(min=0, max=w - 1)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(min=0, max=h - 1)
        if not self.test_mode:
            gt_box[[0, 2]] = gt_box[[0, 2]].clamp(min=0, max=w - 1)
            gt_box[[1, 3]] = gt_box[[1, 3]].clamp(min=0, max=h - 1)

        # assign label to each box by its IoU with gt_box
        if not self.test_mode:
            boxes_ious = bbox_iou_py_vectorized(boxes, gt_box[None]).view(-1)
            label = (boxes_ious > 0.5).float()

        # expression
        exp_tokens = idb['tokens']
        exp_retokens = self.tokenizer.tokenize(' '.join(exp_tokens))
        if flipped:
            exp_retokens = self.flip_tokens(exp_retokens, verbose=True)
        exp_ids = self.tokenizer.convert_tokens_to_ids(exp_retokens)

        if self.test_mode:
            return image, boxes, im_info, exp_ids
        else:
            return image, boxes, im_info, exp_ids, label

    @staticmethod
    def flip_tokens(tokens, verbose=True):
        changed = False
        tokens_new = [tok for tok in tokens]
        for i, tok in enumerate(tokens):
            if tok == 'left':
                tokens_new[i] = 'right'
                changed = True
            elif tok == 'right':
                tokens_new[i] = 'left'
                changed = True
        if verbose and changed:
            logging.info('[Tokens Flip] {} -> {}'.format(tokens, tokens_new))
        return tokens_new

    @staticmethod
    def b64_decode(string):
        return base64.decodebytes(string.encode())

    def load_annotations(self):
        tic = time.time()
        database = []
        db_cache_name = 'refcoco+_boxes_{}_{}'.format(self.boxes, '+'.join(self.image_sets))
        if self.zip_mode:
            db_cache_name = db_cache_name + '_zipmode'
        if self.test_mode:
            db_cache_name = db_cache_name + '_testmode'
        db_cache_root = os.path.join(self.root_path, 'cache')
        db_cache_path = os.path.join(db_cache_root, '{}.pkl'.format(db_cache_name))

        if os.path.exists(db_cache_path):
            if not self.ignore_db_cache:
                # reading cached database
                print('cached database found in {}.'.format(db_cache_path))
                with open(db_cache_path, 'rb') as f:
                    print('loading cached database from {}...'.format(db_cache_path))
                    tic = time.time()
                    database = cPickle.load(f)
                    print('Done (t={:.2f}s)'.format(time.time() - tic))
                    return database
            else:
                print('cached database ignored.')

        # ignore or not find cached database, reload it from annotation file
        print('loading database of split {}...'.format('+'.join(self.image_sets)))
        tic = time.time()

        for ref_id, ref in zip(self.refer_ids, self.refs):
            iset = 'train2014'
            if not self.test_mode:
                gt_x, gt_y, gt_w, gt_h = self.refer.getRefBox(ref_id=ref_id)
            if self.zip_mode:
                image_fn = os.path.join(self.data_path, iset + '.zip@/' + iset,
                                        'COCO_{}_{:012d}.jpg'.format(iset, ref['image_id']))
            else:
                image_fn = os.path.join(self.data_path, iset, 'COCO_{}_{:012d}.jpg'.format(iset, ref['image_id']))
            for sent in ref['sentences']:
                idb = {
                    'sent_id': sent['sent_id'],
                    'ann_id': ref['ann_id'],
                    'ref_id': ref['ref_id'],
                    'image_id': ref['image_id'],
                    'image_fn': image_fn,
                    'width': self.coco.imgs[ref['image_id']]['width'],
                    'height': self.coco.imgs[ref['image_id']]['height'],
                    'raw': sent['raw'],
                    'sent': sent['sent'],
                    'tokens': sent['tokens'],
                    'category_id': ref['category_id'],
                    'gt_box': [gt_x, gt_y, gt_x + gt_w, gt_y + gt_h] if not self.test_mode else None
                }
                database.append(idb)

        print('Done (t={:.2f}s)'.format(time.time() - tic))

        # cache database via cPickle
        if self.cache_db:
            print('caching database to {}...'.format(db_cache_path))
            tic = time.time()
            if not os.path.exists(db_cache_root):
                makedirsExist(db_cache_root)
            with open(db_cache_path, 'wb') as f:
                cPickle.dump(database, f)
            print('Done (t={:.2f}s)'.format(time.time() - tic))

        return database

    @staticmethod
    def group_aspect(database):
        print('grouping aspect...')
        t = time.time()

        # get shape of all images
        widths = torch.as_tensor([idb['width'] for idb in database])
        heights = torch.as_tensor([idb['height'] for idb in database])

        # group
        group_ids = torch.zeros(len(database))
        horz = widths >= heights
        vert = 1 - horz
        group_ids[horz] = 0
        group_ids[vert] = 1

        print('Done (t={:.2f}s)'.format(time.time() - t))

        return group_ids

    def __len__(self):
        return len(self.database)

    def _load_image(self, path):
        if '.zip@' in path:
            return self.zipreader.imread(path).convert('RGB')
        else:
            return Image.open(path).convert('RGB')

    def _load_json(self, path):
        if '.zip@' in path:
            f = self.zipreader.read(path)
            return json.loads(f.decode())
        else:
            with open(path, 'r') as f:
                return json.load(f)

