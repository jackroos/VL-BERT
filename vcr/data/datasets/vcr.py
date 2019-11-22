import os
import time
import jsonlines
import json
import _pickle as cPickle
from PIL import Image
from copy import deepcopy

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

GENDER_NEUTRAL_NAMES = ['Casey', 'Riley', 'Jessie', 'Jackie', 'Avery', 'Jaime', 'Peyton', 'Kerry', 'Jody', 'Kendall',
                        'Frankie', 'Pat', 'Quinn']
# GENDER_NEUTRAL_NAMES = ['person']


class VCRDataset(Dataset):
    def __init__(self, ann_file, image_set, root_path, data_path, transform=None, task='Q2A', test_mode=False,
                 zip_mode=False, cache_mode=False, cache_db=False, ignore_db_cache=True,
                 basic_tokenizer=None, tokenizer=None, pretrained_model_name=None,
                 only_use_relevant_dets=False, add_image_as_a_box=False, mask_size=(14, 14),
                 aspect_grouping=False, basic_align=False, qa2r_noq=False, qa2r_aug=False,
                 seq_len=64,
                 **kwargs):
        """
        Visual Commonsense Reasoning Dataset

        :param ann_file: annotation jsonl file
        :param image_set: image folder name, e.g., 'vcr1images'
        :param root_path: root path to cache database loaded from annotation file
        :param data_path: path to vcr dataset
        :param transform: transform
        :param task: 'Q2A' means question to answer, 'QA2R' means question and answer to rationale,
                     'Q2AR' means question to answer and rationale
        :param test_mode: test mode means no labels available
        :param zip_mode: reading images and metadata in zip archive
        :param cache_mode: cache whole dataset to RAM first, then __getitem__ read them from RAM
        :param ignore_db_cache: ignore previous cached database, reload it from annotation file
        :param tokenizer: default is BertTokenizer from pytorch_pretrained_bert
        :param only_use_relevant_dets: filter out detections not used in query and response
        :param add_image_as_a_box: add whole image as a box
        :param mask_size: size of instance mask of each object
        :param aspect_grouping: whether to group images via their aspect
        :param basic_align: align to tokens retokenized by basic_tokenizer
        :param qa2r_noq: in QA->R, the query contains only the correct answer, without question
        :param qa2r_aug: in QA->R, whether to augment choices to include those with wrong answer in query
        :param kwargs:
        """
        super(VCRDataset, self).__init__()

        assert not cache_mode, 'currently not support cache mode!'
        assert task in ['Q2A', 'QA2R', 'Q2AR'] , 'not support task {}'.format(task)
        assert not qa2r_aug, "Not implemented!"

        self.qa2r_noq = qa2r_noq
        self.qa2r_aug = qa2r_aug

        self.seq_len = seq_len

        categories = ['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                      'trafficlight', 'firehydrant', 'stopsign', 'parkingmeter', 'bench', 'bird', 'cat', 'dog', 'horse',
                      'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                      'suitcase', 'frisbee', 'skis', 'snowboard', 'sportsball', 'kite', 'baseballbat', 'baseballglove',
                      'skateboard', 'surfboard', 'tennisracket', 'bottle', 'wineglass', 'cup', 'fork', 'knife', 'spoon',
                      'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hotdog', 'pizza', 'donut',
                      'cake', 'chair', 'couch', 'pottedplant', 'bed', 'diningtable', 'toilet', 'tv', 'laptop', 'mouse',
                      'remote', 'keyboard', 'cellphone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
                      'clock', 'vase', 'scissors', 'teddybear', 'hairdrier', 'toothbrush']
        self.category_to_idx = {c: i for i, c in enumerate(categories)}
        self.data_path = data_path
        self.root_path = root_path
        self.ann_file = os.path.join(data_path, ann_file)
        self.image_set = image_set
        self.transform = transform
        self.task = task
        self.test_mode = test_mode
        self.zip_mode = zip_mode
        self.cache_mode = cache_mode
        self.cache_db = cache_db
        self.ignore_db_cache = ignore_db_cache
        self.aspect_grouping = aspect_grouping
        self.basic_align = basic_align
        print('Dataset Basic Align: {}'.format(self.basic_align))
        self.cache_dir = os.path.join(root_path, 'cache')
        self.only_use_relevant_dets = only_use_relevant_dets
        self.add_image_as_a_box = add_image_as_a_box
        self.mask_size = mask_size
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

        self.database = self.load_annotations(self.ann_file)
        if self.aspect_grouping:
            assert False, "Not support aspect grouping now!"
            self.group_ids = self.group_aspect(self.database)

        self.person_name_id = 0

    def load_annotations(self, ann_file):
        tic = time.time()
        database = []
        db_cache_name = 'vcr_nometa_{}_{}_{}'.format(self.task, self.image_set, os.path.basename(ann_file)[:-len('.jsonl')])
        if self.only_use_relevant_dets:
            db_cache_name = db_cache_name + '_only_relevant_dets'
        if self.zip_mode:
            db_cache_name = db_cache_name + '_zipped'
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
        print('loading database from {}...'.format(ann_file))
        tic = time.time()

        with jsonlines.open(ann_file) as reader:
            for ann in reader:
                if self.zip_mode:
                    img_fn = os.path.join(self.data_path, self.image_set + '.zip@/' + self.image_set, ann['img_fn'])
                    metadata_fn = os.path.join(self.data_path, self.image_set + '.zip@/' + self.image_set, ann['metadata_fn'])
                else:
                    img_fn = os.path.join(self.data_path, self.image_set, ann['img_fn'])
                    metadata_fn = os.path.join(self.data_path, self.image_set, ann['metadata_fn'])

                db_i = {
                    'annot_id': ann['annot_id'],
                    'objects': ann['objects'],
                    'img_fn': img_fn,
                    'metadata_fn': metadata_fn,
                    'question': ann['question'],
                    'answer_choices': ann['answer_choices'],
                    'answer_label': ann['answer_label'] if not self.test_mode else None,
                    'rationale_choices': ann['rationale_choices'],
                    'rationale_label': ann['rationale_label'] if not self.test_mode else None,
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

    def retokenize_and_convert_to_ids_with_tag(self, tokens, objects_replace_name, non_obj_tag=-1):
        parsed_tokens = []
        tags = []
        align_ids = []
        raw = []
        align_id = 0
        for mixed_token in tokens:
            if isinstance(mixed_token, list):
                tokens = [objects_replace_name[o] for o in mixed_token]
                retokenized_tokens = self.tokenizer.tokenize(tokens[0])
                raw.append(tokens[0])
                tags.extend([mixed_token[0] + non_obj_tag + 1 for _ in retokenized_tokens])
                align_ids.extend([align_id for _ in retokenized_tokens])
                align_id += 1
                for token, o in zip(tokens[1:], mixed_token[1:]):
                    retokenized_tokens.append('and')
                    tags.append(non_obj_tag)
                    align_ids.append(align_id)
                    align_id += 1
                    re_tokens = self.tokenizer.tokenize(token)
                    retokenized_tokens.extend(re_tokens)
                    tags.extend([o + non_obj_tag + 1 for _ in re_tokens])
                    align_ids.extend([align_id for _ in re_tokens])
                    align_id += 1
                    raw.extend(['and', token])
                parsed_tokens.extend(retokenized_tokens)
            else:
                if self.basic_align:
                    # basic align
                    basic_tokens = self.basic_tokenizer.tokenize(mixed_token)
                    raw.extend(basic_tokens)
                    for t in basic_tokens:
                        retokenized_tokens = self.tokenizer.tokenize(t)
                        parsed_tokens.extend(retokenized_tokens)
                        align_ids.extend([align_id for _ in retokenized_tokens])
                        tags.extend([non_obj_tag for _ in retokenized_tokens])
                        align_id += 1
                else:
                    # fully align to original tokens
                    raw.append(mixed_token)
                    retokenized_tokens = self.tokenizer.tokenize(mixed_token)
                    parsed_tokens.extend(retokenized_tokens)
                    align_ids.extend([align_id for _ in retokenized_tokens])
                    tags.extend([non_obj_tag for _ in retokenized_tokens])
                    align_id += 1
        ids = self.tokenizer.convert_tokens_to_ids(parsed_tokens)
        ids_with_tag = list(zip(ids, tags, align_ids))
        
        return ids_with_tag, raw

    @staticmethod
    def keep_only_relevant_dets(question, answer_choices, rationale_choices):
        dets_to_use = []
        for i, tok in enumerate(question):
            if isinstance(tok, list):
                for j, o in enumerate(tok):
                    if o not in dets_to_use:
                        dets_to_use.append(o)
                    question[i][j] = dets_to_use.index(o)
        if answer_choices is not None:
            for n, answer in enumerate(answer_choices):
                for i, tok in enumerate(answer):
                    if isinstance(tok, list):
                        for j, o in enumerate(tok):
                            if o not in dets_to_use:
                                dets_to_use.append(o)
                            answer_choices[n][i][j] = dets_to_use.index(o)
        if rationale_choices is not None:
            for n, rationale in enumerate(rationale_choices):
                for i, tok in enumerate(rationale):
                    if isinstance(tok, list):
                        for j, o in enumerate(tok):
                            if o not in dets_to_use:
                                dets_to_use.append(o)
                            rationale_choices[n][i][j] = dets_to_use.index(o)

        return dets_to_use, question, answer_choices, rationale_choices

    def __getitem__(self, index):
        # self.person_name_id = 0
        idb = deepcopy(self.database[index])

        metadata = self._load_json(idb['metadata_fn'])
        idb['boxes'] = metadata['boxes']
        idb['segms'] = metadata['segms']
        # idb['width'] = metadata['width']
        # idb['height'] = metadata['height']
        if self.only_use_relevant_dets:
            dets_to_use, idb['question'], idb['answer_choices'], idb['rationale_choices'] = \
                self.keep_only_relevant_dets(idb['question'],
                                             idb['answer_choices'],
                                             idb['rationale_choices'] if not self.task == 'Q2A' else None)
            idb['objects'] = [idb['objects'][i] for i in dets_to_use]
            idb['boxes'] = [idb['boxes'][i] for i in dets_to_use]
            idb['segms'] = [idb['segms'][i] for i in dets_to_use]
        objects_replace_name = []
        for o in idb['objects']:
            if o == 'person':
                objects_replace_name.append(GENDER_NEUTRAL_NAMES[self.person_name_id])
                self.person_name_id = (self.person_name_id + 1) % len(GENDER_NEUTRAL_NAMES)
            else:
                objects_replace_name.append(o)

        non_obj_tag = 0 if self.add_image_as_a_box else -1
        idb['question'] = self.retokenize_and_convert_to_ids_with_tag(idb['question'],
                                                                      objects_replace_name=objects_replace_name,
                                                                      non_obj_tag=non_obj_tag)

        idb['answer_choices'] = [self.retokenize_and_convert_to_ids_with_tag(answer,
                                                                             objects_replace_name=objects_replace_name,
                                                                             non_obj_tag=non_obj_tag)
                                 for answer in idb['answer_choices']]

        idb['rationale_choices'] = [self.retokenize_and_convert_to_ids_with_tag(rationale,
                                                                                objects_replace_name=objects_replace_name,
                                                                                non_obj_tag=non_obj_tag)
                                    for rationale in idb['rationale_choices']] if not self.task == 'Q2A' else None

        # truncate text to seq_len
        if self.task == 'Q2A':
            q = idb['question'][0]
            for a, a_raw in idb['answer_choices']:
                while len(q) + len(a) > self.seq_len:
                    if len(a) > len(q):
                        a.pop()
                    else:
                        q.pop()
        elif self.task == 'QA2R':
            if not self.test_mode:
                q = idb['question'][0]
                a = idb['answer_choices'][idb['answer_label']][0]
                for r, r_raw in idb['rationale_choices']:
                    while len(q) + len(a) + len(r) > self.seq_len:
                        if len(r) > (len(q) + len(a)):
                            r.pop()
                        elif len(q) > 1:
                            q.pop()
                        else:
                            a.pop()
        else:
            raise NotImplemented

        image = self._load_image(idb['img_fn'])
        w0, h0 = image.size
        objects = idb['objects']

        # extract bounding boxes and instance masks in metadata
        boxes = torch.zeros((len(objects), 6))
        masks = torch.zeros((len(objects), *self.mask_size))
        if len(objects) > 0:
            boxes[:, :5] = torch.tensor(idb['boxes'])
            boxes[:, 5] = torch.tensor([self.category_to_idx[obj] for obj in objects])
            for i in range(len(objects)):
                seg_polys = [torch.as_tensor(seg) for seg in idb['segms'][i]]
                masks[i] = generate_instance_mask(seg_polys, idb['boxes'][i], mask_size=self.mask_size,
                                                  dtype=torch.float32, copy=False)
        if self.add_image_as_a_box:
            image_box = torch.as_tensor([[0, 0, w0 - 1, h0 - 1, 1.0, 0]])
            image_mask = torch.ones((1, *self.mask_size))
            boxes = torch.cat((image_box, boxes), dim=0)
            masks = torch.cat((image_mask, masks), dim=0)

        question, question_raw = idb['question']
        question_align_matrix = get_align_matrix([w[2] for w in question])
        answer_choices, answer_choices_raw = zip(*idb['answer_choices'])
        answer_choices = list(answer_choices)
        answer_align_matrix = [get_align_matrix([w[2] for w in a]) for a in answer_choices]
        answer_label = torch.as_tensor(idb['answer_label']) if not self.test_mode else None
        if not self.task == 'Q2A':
            rationale_choices = [r[0] for r in idb['rationale_choices']]
            rationale_align_matrix = [get_align_matrix([w[2] for w in r]) for r in rationale_choices]
            rationale_label = torch.as_tensor(idb['rationale_label']) if not self.test_mode else None

        # transform
        im_info = torch.tensor([w0, h0, 1.0, 1.0, index])
        if self.transform is not None:
            image, boxes, masks, im_info = self.transform(image, boxes, masks, im_info)

        # clamp boxes
        w = im_info[0].item()
        h = im_info[1].item()
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(min=0, max=w - 1)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(min=0, max=h - 1)

        if self.task == 'Q2AR':
            if not self.test_mode:
                outputs = (image, boxes, masks,
                           question, question_align_matrix,
                           answer_choices, answer_align_matrix, answer_label,
                           rationale_choices, rationale_align_matrix, rationale_label,
                           im_info)
            else:
                outputs = (image, boxes, masks,
                           question, question_align_matrix,
                           answer_choices, answer_align_matrix,
                           rationale_choices, rationale_align_matrix,
                           im_info)
        elif self.task == 'Q2A':
            if not self.test_mode:
                outputs = (image, boxes, masks,
                           question, question_align_matrix,
                           answer_choices, answer_align_matrix, answer_label,
                           im_info)
            else:
                outputs = (image, boxes, masks,
                           question, question_align_matrix,
                           answer_choices, answer_align_matrix,
                           im_info)
        elif self.task == 'QA2R':
            if not self.test_mode:
                outputs = (image, boxes, masks,
                           ([] if self.qa2r_noq else question) + answer_choices[answer_label],
                           answer_align_matrix[answer_label] if self.qa2r_noq else block_digonal_matrix(question_align_matrix, answer_align_matrix[answer_label]),
                           rationale_choices, rationale_align_matrix, rationale_label,
                           im_info)
            else:
                outputs = (image, boxes, masks,
                           [([] if self.qa2r_noq else question) + a for a in answer_choices],
                           [m if self.qa2r_noq else block_digonal_matrix(question_align_matrix, m)
                            for m in answer_align_matrix],
                           rationale_choices, rationale_align_matrix,
                           im_info)

        return outputs

    def __len__(self):
        return len(self.database)

    def _load_image(self, path):
        if '.zip@' in path:
            return self.zipreader.imread(path)
        else:
            return Image.open(path)

    def _load_json(self, path):
        if '.zip@' in path:
            f = self.zipreader.read(path)
            return json.loads(f.decode())
        else:
            with open(path, 'r') as f:
                return json.load(f)

    @property
    def data_names(self):
        if not self.test_mode:
            if self.task == 'Q2A':
                data_names = ['image', 'boxes', 'masks',
                              'question', 'question_align_matrix',
                              'answer_choices', 'answer_align_matrix', 'answer_label',
                              'im_info']
            elif self.task == 'QA2R':
                data_names = ['image', 'boxes', 'masks',
                              'question', 'question_align_matrix',
                              'rationale_choices', 'rationale_align_matrix', 'rationale_label',
                              'im_info']
            else:
                data_names = ['image', 'boxes', 'masks',
                              'question', 'question_align_matrix',
                              'answer_choices', 'answer_align_matrix', 'answer_label',
                              'rationale_choices', 'rationale_align_matrix', 'rationale_label',
                              'im_info']
        else:
            if self.task == 'Q2A':
                data_names = ['image', 'boxes', 'masks',
                              'question', 'question_align_matrix',
                              'answer_choices', 'answer_align_matrix',
                              'im_info']
            elif self.task == 'QA2R':
                data_names = ['image', 'boxes', 'masks',
                              'question', 'question_align_matrix',
                              'rationale_choices', 'rationale_align_matrix',
                              'im_info']
            else:
                data_names = ['image', 'boxes', 'masks',
                              'question', 'question_align_matrix',
                              'answer_choices', 'answer_align_matrix',
                              'rationale_choices', 'rationale_align_matrix',
                              'im_info']

        return data_names
