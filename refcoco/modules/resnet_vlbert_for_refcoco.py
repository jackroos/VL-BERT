import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from external.pytorch_pretrained_bert import BertTokenizer
from common.module import Module
from common.fast_rcnn import FastRCNN
from common.visual_linguistic_bert import VisualLinguisticBert, VisualLinguisticBertMVRCHeadTransform

BERT_WEIGHTS_NAME = 'pytorch_model.bin'


class ResNetVLBERT(Module):
    def __init__(self, config):

        super(ResNetVLBERT, self).__init__(config)

        self.image_feature_extractor = FastRCNN(config,
                                                average_pool=True,
                                                final_dim=config.NETWORK.IMAGE_FINAL_DIM,
                                                enable_cnn_reg_loss=False)
        self.object_linguistic_embeddings = nn.Embedding(1, config.NETWORK.VLBERT.hidden_size)
        self.image_feature_bn_eval = config.NETWORK.IMAGE_FROZEN_BN
        self.tokenizer = BertTokenizer.from_pretrained(config.NETWORK.BERT_MODEL_NAME)

        language_pretrained_model_path = None
        if config.NETWORK.BERT_PRETRAINED != '':
            language_pretrained_model_path = '{}-{:04d}.model'.format(config.NETWORK.BERT_PRETRAINED,
                                                                      config.NETWORK.BERT_PRETRAINED_EPOCH)
        elif os.path.isdir(config.NETWORK.BERT_MODEL_NAME):
            weight_path = os.path.join(config.NETWORK.BERT_MODEL_NAME, BERT_WEIGHTS_NAME)
            if os.path.isfile(weight_path):
                language_pretrained_model_path = weight_path
        self.language_pretrained_model_path = language_pretrained_model_path
        if language_pretrained_model_path is None:
            print("Warning: no pretrained language model found, training from scratch!!!")

        self.vlbert = VisualLinguisticBert(config.NETWORK.VLBERT,
                                         language_pretrained_model_path=language_pretrained_model_path)

        transform = VisualLinguisticBertMVRCHeadTransform(config.NETWORK.VLBERT)
        linear = nn.Linear(config.NETWORK.VLBERT.hidden_size, 1)
        self.final_mlp = nn.Sequential(
            transform,
            nn.Dropout(config.NETWORK.CLASSIFIER_DROPOUT, inplace=False),
            linear
        )

        # init weights
        self.init_weight()

        self.fix_params()

    def init_weight(self):
        self.image_feature_extractor.init_weight()
        if self.object_linguistic_embeddings is not None:
            self.object_linguistic_embeddings.weight.data.normal_(mean=0.0,
                                                                  std=self.config.NETWORK.VLBERT.initializer_range)
        for m in self.final_mlp.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.constant_(m.bias, 0)

    def train(self, mode=True):
        super(ResNetVLBERT, self).train(mode)
        # turn some frozen layers to eval mode
        if self.image_feature_bn_eval:
            self.image_feature_extractor.bn_eval()

    def fix_params(self):
        pass

    def train_forward(self,
                      image,
                      boxes,
                      im_info,
                      expression,
                      label,
                      ):
        ###########################################

        # visual feature extraction
        images = image
        box_mask = (boxes[:, :, 0] > - 1.5)
        max_len = int(box_mask.sum(1).max().item())
        origin_len = boxes.shape[1]
        box_mask = box_mask[:, :max_len]
        boxes = boxes[:, :max_len]
        label = label[:, :max_len]

        obj_reps = self.image_feature_extractor(images=images,
                                                boxes=boxes,
                                                box_mask=box_mask,
                                                im_info=im_info,
                                                classes=None,
                                                segms=None)

        ############################################
        # prepare text
        cls_id, sep_id = self.tokenizer.convert_tokens_to_ids(['[CLS]', '[SEP]'])
        text_input_ids = expression.new_zeros((expression.shape[0], expression.shape[1] + 2))
        text_input_ids[:, 0] = cls_id
        text_input_ids[:, 1:-1] = expression
        _sep_pos = (text_input_ids > 0).sum(1)
        _batch_inds = torch.arange(expression.shape[0], device=expression.device)
        text_input_ids[_batch_inds, _sep_pos] = sep_id
        text_token_type_ids = text_input_ids.new_zeros(text_input_ids.shape)
        text_mask = text_input_ids > 0
        text_visual_embeddings = obj_reps['obj_reps'][:, 0].unsqueeze(1).repeat((1, text_input_ids.shape[1], 1))

        object_linguistic_embeddings = self.object_linguistic_embeddings(
            boxes.new_zeros((boxes.shape[0], boxes.shape[1])).long()
        )
        object_vl_embeddings = torch.cat((obj_reps['obj_reps'], object_linguistic_embeddings), -1)

        ###########################################

        # Visual Linguistic BERT

        hidden_states_text, hidden_states_regions, _ = self.vlbert(text_input_ids,
                                                                 text_token_type_ids,
                                                                 text_visual_embeddings,
                                                                 text_mask,
                                                                 object_vl_embeddings,
                                                                 box_mask,
                                                                 output_all_encoded_layers=False,
                                                                 output_text_and_object_separately=True)

        ###########################################
        outputs = {}

        # classifier
        logits = self.final_mlp(hidden_states_regions).squeeze(-1)

        # loss
        cls_loss = F.binary_cross_entropy_with_logits(logits[box_mask], label[box_mask])

        # pad back to origin len for compatibility with DataParallel
        logits_ = logits.new_zeros((logits.shape[0], origin_len)).fill_(-10000.0)
        logits_[:, :logits.shape[1]] = logits
        logits = logits_
        label_ = label.new_zeros((logits.shape[0], origin_len)).fill_(-1)
        label_[:, :label.shape[1]] = label
        label = label_

        outputs.update({'label_logits': logits,
                        'label': label,
                        'cls_loss': cls_loss})

        loss = cls_loss.mean()

        return outputs, loss

    def inference_forward(self,
                          image,
                          boxes,
                          im_info,
                          expression):

        ###########################################

        # visual feature extraction
        images = image
        box_mask = (boxes[:, :, 0] > - 1.5)
        max_len = int(box_mask.sum(1).max().item())
        origin_len = boxes.shape[1]
        box_mask = box_mask[:, :max_len]
        boxes = boxes[:, :max_len]

        obj_reps = self.image_feature_extractor(images=images,
                                                boxes=boxes,
                                                box_mask=box_mask,
                                                im_info=im_info,
                                                classes=None,
                                                segms=None)

        ############################################
        # prepare text
        cls_id, sep_id = self.tokenizer.convert_tokens_to_ids(['[CLS]', '[SEP]'])
        text_input_ids = expression.new_zeros((expression.shape[0], expression.shape[1] + 2))
        text_input_ids[:, 0] = cls_id
        text_input_ids[:, 1:-1] = expression
        _sep_pos = (text_input_ids > 0).sum(1)
        _batch_inds = torch.arange(expression.shape[0], device=expression.device)
        text_input_ids[_batch_inds, _sep_pos] = sep_id
        text_token_type_ids = text_input_ids.new_zeros(text_input_ids.shape)
        text_mask = text_input_ids > 0
        text_visual_embeddings = obj_reps['obj_reps'][:, 0].unsqueeze(1).repeat((1, text_input_ids.shape[1], 1))

        object_linguistic_embeddings = self.object_linguistic_embeddings(
            boxes.new_zeros((boxes.shape[0], boxes.shape[1])).long()
        )
        object_vl_embeddings = torch.cat((obj_reps['obj_reps'], object_linguistic_embeddings), -1)

        ###########################################

        # Visual Linguistic BERT

        hidden_states_text, hidden_states_regions, _ = self.vlbert(text_input_ids,
                                                                 text_token_type_ids,
                                                                 text_visual_embeddings,
                                                                 text_mask,
                                                                 object_vl_embeddings,
                                                                 box_mask,
                                                                 output_all_encoded_layers=False,
                                                                 output_text_and_object_separately=True)

        ###########################################
        outputs = {}

        # classifier
        logits = self.final_mlp(hidden_states_regions).squeeze(-1)

        # pad back to origin len for compatibility with DataParallel
        logits_ = logits.new_zeros((logits.shape[0], origin_len)).fill_(-10000.0)
        logits_[:, :logits.shape[1]] = logits
        logits = logits_

        w_ratio = im_info[:, 2]
        h_ratio = im_info[:, 3]
        pred_boxes = boxes[_batch_inds, logits.argmax(1), :4]
        pred_boxes[:, [0, 2]] /= w_ratio.unsqueeze(1)
        pred_boxes[:, [1, 3]] /= h_ratio.unsqueeze(1)
        outputs.update({'label_logits': logits,
                        'pred_boxes': pred_boxes})

        return outputs
