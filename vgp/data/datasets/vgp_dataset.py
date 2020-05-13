import os
import time
import _pickle as cPickle
from PIL import Image
from copy import deepcopy
import numpy as np
import pandas as pd
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
from common.nlp.roberta import RobertaTokenizer
from vgp.function.hard_neg_sampling import main



class VGPDataset(Dataset):
    def __init__(self, captions_set, ann_file, roi_set, image_set, root_path, data_path, small_version=False,
                 negative_sampling='hard', transform=None, test_mode=False, zip_mode=False, cache_mode=False,
                 cache_db=False, ignore_db_cache=True, basic_tokenizer=None, tokenizer=None, pretrained_model_name=None,
                 add_image_as_a_box=True, **kwargs):
        """
        Visual Grounded Paraphrase Dataset

        :param ann_file: annotation csv file
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
        self.captions_set = os.path.join(data_path, captions_set)
        self.ann_file = os.path.join(data_path, ann_file)
        self.roi_set = os.path.join(data_path, roi_set)
        self.image_set = os.path.join(self.data_path, image_set)
        self.small = small_version
        self.neg_sampling = negative_sampling
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

        self.database = self.load_captions(self.captions_set)

    def load_captions(self, captions_set):
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
        print('loading database from {} and creating pairs...'.format(captions_set))
        tic = time.time()
        if self.neg_sampling == "hard":
            path_similarities = os.path.join(self.captions_set, "similarities.csv")
            if not os.path.exists(path_similarities):
                print("It seems hard negative mining has not been done for this set of captions, run it now")
                model_path = os.path.join(os.getcwd(), "model/pretrained_model/resnet101-pt-vgbua-0000.model")
                main(self.captions_set, self.image_set, model_path, batch_size=4, n_neighbors=20, use_saved=True)
            similarities_df = pd.read_csv(path_similarities)
        img_id_list = np.array(os.listdir(captions_set))
        for k, folder in enumerate(img_id_list):
            if folder.endswith(".txt"):
                img_id = folder[:-4]
                path = os.path.join(captions_set, folder)
                list_captions = open(path).read().split("\n")[:-1]

                if self.small:
                    positive_captions = np.random.choice(list_captions, 2, replace=False)
                    n_negative = 1
                else:
                    positive_captions = list_captions
                    n_negative = 2
                # Create pairs of captions that describe the same image
                for i in range(len(positive_captions)):
                    for j in range(i):
                        # create a unique id for each instance in the data set
                        pair_id = "{}_{}_{}".format(str(k), str(i), str(j))
                        db_i = {
                            'pair_id': pair_id,
                            'img_id': img_id,
                            'caption1': list_captions[i],
                            'caption2': list_captions[j],
                            'label': 0
                        }
                        database.append(db_i)

                # Select one or two negative captions
                if self.neg_sampling == 'random':
                    other_imgs = img_id_list[img_id_list != folder]
                    # Fix the seed to have data set reproducibility
                    np.random.seed(k)
                    neg_image = np.random.choice(other_imgs, size=1)[0]
                    np.random.seed(k)
                    neg_path = os.path.join(captions_set, neg_image)
                else:
                    if self.neg_sampling != "hard":
                        print("{} negative sampling is not supported, hard negative sampling will "
                              "be used".format(self.neg_sampling))
                    similar_img_idx = similarities_df[similarities_df["img_id"] == int(img_id)]["2"].values[0]
                    neg_img = similarities_df.iloc[similar_img_idx]["img_id"]
                    neg_path = os.path.join(captions_set, str(neg_img) + ".txt")

                # Create negative pairs
                neg_captions = np.random.choice(open(neg_path).read().split("\n")[:-1], size=n_negative,
                                                replace=False)
                for idx, caption in enumerate(positive_captions):
                    # if we want the small data set only create one negative pair
                    if self.small and idx > 0:
                        break
                    else:
                        for idx_bis, wrong_caption in enumerate(neg_captions):
                            # Randomly flip whether the wrong caption comes first or second, fix the seed for every image
                            np.random.seed(k + idx + idx_bis)
                            flip = np.random.randint(2, size=1).astype(bool)[0]
                            pair_id = "{}_{}_{}".format(str(k), str(idx), str(idx_bis + len(positive_captions)))
                            db_i = {
                                'pair_id': pair_id,
                                'img_id': img_id,
                                'label': 1 + flip
                            }
                            if flip:
                                db_i['caption1'] = wrong_caption
                                db_i['caption2'] = caption
                            else:
                                db_i['caption1'] = caption
                                db_i['caption2'] = wrong_caption
                            database.append(db_i)
            else:
                continue
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

        # Load image and regions of interest
        img_id = idb['img_id']
        image = self._load_image(img_id)
        boxes = self._load_roi(img_id)
        w0, h0 = image.size

        boxes = torch.tensor(boxes)
        if self.add_image_as_a_box:
            image_box = torch.as_tensor([[0, 0, w0 - 1, h0 - 1, 0]])
            boxes = torch.cat((image_box.type(torch.float32), boxes.type(torch.float32)), dim=0)

        # transform
        im_info = torch.tensor([w0, h0, 1.0, 1.0, index])
        if self.transform is not None:
            image, boxes, _, im_info = self.transform(image, boxes, None, im_info)

        # clamp boxes
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(min=0, max=w0 - 1)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(min=0, max=h0 - 1)
        
        # keep box names and dimensions separated
        box_names = boxes[:, -1].tolist()
        boxes = boxes[:, :4]

        # Format input text
        captions = [idb["caption1"], idb["caption2"]]
        formatted_text, relevant_boxes = self._extract_text_and_roi(captions)

        # Tokenize and align tokens with visual boxes
        tokens1 = [self.tokenizer.tokenize(raw_text) for raw_text in formatted_text[0]]
        txt_visual_ground1 = [[relevant_boxes[0][i]]*len(tokens) for i, tokens in enumerate(tokens1)]
        tokens2 = [self.tokenizer.tokenize(raw_text) for raw_text in formatted_text[1]]
        txt_visual_ground2 = [[relevant_boxes[1][i]] * len(tokens) for i, tokens in enumerate(tokens2)]

        # Flatten lists
        tokens1 = [token for sublist in tokens1 for token in sublist]
        tokens2 = [token for sublist in tokens2 for token in sublist]
        txt_visual_ground1 = [box for sublist in txt_visual_ground1 for box in sublist]
        txt_visual_ground2 = [box for sublist in txt_visual_ground2 for box in sublist]

        # Convert token to ids and concatenate visual grounding
        caption1_ids = torch.as_tensor(self.tokenizer.convert_tokens_to_ids(tokens1)).unsqueeze(1)
        caption2_ids = torch.as_tensor(self.tokenizer.convert_tokens_to_ids(tokens2)).unsqueeze(1)
        vl_ground_idx1 = torch.as_tensor([box_names.index(box_id) if box_id in box_names else box_names.index(0)
                                          for box_id in txt_visual_ground1]).unsqueeze(1)
        vl_ground_idx2 = torch.as_tensor([box_names.index(box_id) if box_id in box_names else box_names.index(0)
                                          for box_id in txt_visual_ground2]).unsqueeze(1)
        final_input_1 = torch.cat((caption1_ids, vl_ground_idx1), dim=1)
        final_input_2 = torch.cat((caption2_ids, vl_ground_idx2), dim=1)

        # Add (later) mask to locate sub-phrases inside full sentence

        # Load label
        label = torch.as_tensor(int(idb['label'])) if not self.test_mode else None

        if not self.test_mode:
            outputs = (image, boxes, final_input_1, final_input_2, im_info, label)
        else:
            outputs = (image, boxes, final_input_1, final_input_2, im_info)

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
                boxes.append([dimensions['xmin'], dimensions['ymin'], dimensions['xmax'], dimensions['ymax'],
                              int(obj[0].text)])
        return boxes

    def _extract_text_and_roi(self, captions):
        formatted_text = []
        relevant_boxes = []
        for caption in captions:
            caption = caption.replace("[", "¥¥¥").replace("]", "¥¥¥").split("¥¥¥")
            extracted_entities = []
            box_ids = []
            for string in caption:
                if string.startswith("/EN#"):
                    box_id = string.split("#")[1].split("/")[0]
                    text = string.split(" ")[1:]
                    if len(text) == 1:
                        text = text[0]
                    else:
                        text = " ".join(text)
                else:
                    box_id = "0"
                    text = string
                extracted_entities.append(text)
                try:
                    box_ids.append(int(box_id))
                except:
                    print(box_id)
                    print(captions)
                    assert False
            formatted_text.append(extracted_entities)
            relevant_boxes.append(box_ids)
        return formatted_text, relevant_boxes

    @property
    def data_names(self):
        if not self.test_mode:
            data_names = ['image', 'boxes', 'caption1', 'caption2', 'im_info', 'label']
        else:
            data_names = ['image', 'boxes', 'caption1', 'caption2', 'im_info']

        return data_names


def test_vgp():
    ann_file = "full_data_type_phrase_pair_train.csv"
    image_set = "flickr30k-images"
    roi_set = "Annotations"
    root_path = ""
    data_path = os.path.join(os.getcwd(), "data/vgp/")
    dataset = VGPDataset(captions_set="train_captions", ann_file=ann_file, roi_set=roi_set, image_set=image_set,
                         small_version=True, negative_sampling='hard', root_path=root_path, data_path=data_path)
    print(len(dataset.__getitem__(0)))


if __name__ == "__main__":
    test_vgp()
