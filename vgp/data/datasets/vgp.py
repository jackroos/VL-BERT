import os
import time
import _pickle as cPickle
from PIL import Image
from copy import deepcopy
from csv import DictReader
import xml.etree.ElementTree as ET

import sys
root_path = os.path.abspath(os.getcwd())
if root_path not in sys.path:
    sys.path.append(root_path)

import torch
from torch.utils.data import Dataset
from external.pytorch_pretrained_bert import BertTokenizer, BasicTokenizer

from common.utils.zipreader import ZipReader
from common.utils.create_logger import makedirsExist
from common.utils.mask import generate_instance_mask
from common.nlp.misc import get_align_matrix
from common.utils.misc import block_digonal_matrix
from common.nlp.misc import random_word_with_token_ids
from common.nlp.roberta import RobertaTokenizer


class VGPDataset(Dataset):
    def __init__(self, full_sentences_file, paraphrase_file, roi_set, image_set, root_path, data_path, transform=None,
                 test_mode=False, zip_mode=False, cache_mode=False, cache_db=False, ignore_db_cache=True,
                 basic_tokenizer=None, tokenizer=None, pretrained_model_name=None, add_image_as_a_box=False, **kwargs):
        """
        Visual Grounded Paraphrase Dataset

        :param ann_file: annotation jsonl file
        :param image_set: image folder name, e.g., 'vcr1images'
        :param root_path: root path to cache database loaded from annotation file
        :param data_path: path to vcr dataset
        :param transform: transform
        :param test_mode: test mode means no labels available
        :param cache_mode: cache whole dataset to RAM first, then __getitem__ read them from RAM
        :param ignore_db_cache: ignore previous cached database, reload it from annotation file
        :param tokenizer: default is BertTokenizer from pytorch_pretrained_bert
        :param add_image_as_a_box: add whole image as a box
        :param kwargs:
        """
        super(VGPDataset, self).__init__()

        assert not cache_mode, 'currently not support cache mode!'

        self.data_path = data_path
        self.root_path = root_path
        self.full_sentences_file = os.path.join(data_path, full_sentences_file)
        self.paraphrase_file = os.path.join(data_path, paraphrase_file)
        self.roi_set = os.path.join(data_path, roi_set)
        self.image_set = os.path.join(self.data_path, image_set)
        self.transform = transform
        self.test_mode = test_mode
        self.zip_mode = zip_mode
        self.cache_mode = cache_mode
        self.cache_db = cache_db
        self.ignore_db_cache = ignore_db_cache
        self.cache_dir = os.path.join(root_path, 'cache')
        self.add_image_as_a_box = add_image_as_a_box
        if not os.path.exists(self.cache_dir):
            makedirsExist(self.cache_dir)
        self.basic_tokenizer = basic_tokenizer if basic_tokenizer is not None \
            else BasicTokenizer(do_lower_case=True)
        if tokenizer is None:
            if pretrained_model_name is None:
                pretrained_model_name = 'bert-base-uncased'
            if 'roberta' in pretrained_model_name:
                tokenizer = RobertaTokenizer.from_pretrained(pretrained_model_name, cache_dir=self.cache_dir)
            else:
                tokenizer = BertTokenizer.from_pretrained(pretrained_model_name, cache_dir=self.cache_dir)
        self.tokenizer = tokenizer

        if zip_mode:
            self.zipreader = ZipReader()

        self.database = self.load_phrases(self.paraphrase_file)

    def load_phrases(self, paraphrase_file):
        database = []
        db_cache_name = 'vgp_nometa'
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
        print('loading database from {}...'.format(paraphrase_file))
        tic = time.time()

        # open file in read mode
        with open(paraphrase_file, 'r') as read_obj:
            csv_reader = DictReader(read_obj)
            # Iterate over each row in the csv using reader object
            for paraph in csv_reader:
                db_i = {
                    'img_id': paraph["image"],
                    'phrase1': paraph['original_phrase1'],
                    'phrase2': paraph['original_phrase2'],
                    'label': paraph['type_label']
                }
                database.append(db_i)
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

    def __getitem__(self, index):
        idb = deepcopy(self.database[index])

        img_id = idb['img_id']
        image = self._load_image(img_id)
        boxes = self._load_roi(img_id)

        # Format input text
        phrase1_tokens = self.tokenizer.tokenize(idb['phrase1'])
        phrase1_ids = torch.as_tensor(self.tokenizer.convert_tokens_to_ids(phrase1_tokens)).unsqueeze(0)
        phrase2_tokens = self.tokenizer.tokenize(idb['phrase2'])
        phrase2_ids = torch.as_tensor(self.tokenizer.convert_tokens_to_ids(phrase2_tokens)).unsqueeze(0)

        # Add mask to locate sub-phrases inside full sentence
        # For now full sentence is just the sub-phrase
        phrase1_mask = torch.ones_like(phrase1_ids)
        phrase2_mask = torch.ones_like(phrase2_ids)
        sentence1 = torch.cat((phrase1_ids, phrase1_mask), dim=0)
        sentence2 = torch.cat((phrase2_ids, phrase2_mask), dim=0)

        w0, h0 = image.size

        # extract bounding boxes and instance masks in metadata
        boxes = torch.tensor(boxes)
        if self.add_image_as_a_box:
            image_box = torch.as_tensor([[0, 0, w0 - 1, h0 - 1]])
            boxes = torch.cat((image_box, boxes), dim=0)
            
        # transform
        im_info = torch.tensor([w0, h0, 1.0, 1.0, index])
        if self.transform is not None:
            image, boxes, _, im_info = self.transform(image, boxes, None, im_info)

        # clamp boxes
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(min=0, max=w0 - 1)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(min=0, max=h0 - 1)

        # Load label
        label = torch.as_tensor(int(idb['label'])) if not self.test_mode else None

        if not self.test_mode:
            outputs = (image, boxes, sentence1, sentence2, label, im_info)
        else:
            outputs = (image, boxes, sentence1, sentence2, im_info)

        return outputs

    def __len__(self):
        return len(self.database)

    def _load_image(self, img_id):
        path = os.path.join(self.image_set, img_id + ".jpg")
        return Image.open(path)

    def _load_roi(self, img_id):
        path = os.path.join(self.roi_set, img_id + '.xml')
        boxes = []
        parsed_tree = ET.parse(path)
        root = parsed_tree.getroot()
        for obj in root[2:]:
            if obj[1].tag == 'bndbox':
                dimensions = {obj[1][dim].tag: int(obj[1][dim].text) for dim in range(4)}
                boxes.append([dimensions['xmin'], dimensions['ymin'], dimensions['xmax'], dimensions['ymax']])
        return boxes

    @property
    def data_names(self):
        if not self.test_mode:
            data_names = ['image', 'boxes', 'sentence1', 'sentence2', 'label', 'im_info']
        else:
            data_names = ['image', 'boxes', 'sentence1', 'sentence2', 'im_info']

        return data_names


def test_vgp():
    paraphrase_file = "full_data_type_phrase_pair_train.csv"
    image_set = "flickr30k-images"
    roi_set = "Annotations"
    root_path = ""
    data_path = os.path.join(os.getcwd(), "data/vgp/")
    dataset = VGPDataset(full_sentences_file="", paraphrase_file=paraphrase_file, roi_set=roi_set, image_set=image_set, root_path=root_path,
                         data_path=data_path)
    print(len(dataset.__getitem__(0)))


if __name__ == "__main__":
    test_vgp()
