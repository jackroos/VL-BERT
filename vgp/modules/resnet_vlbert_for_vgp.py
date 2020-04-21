import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from external.pytorch_pretrained_bert import BertTokenizer
from common.module import Module
from common.fast_rcnn import FastRCNN
from common.nlp.time_distributed import TimeDistributed
from common.visual_linguistic_bert import VisualLinguisticBert, VisualLinguisticBertMVRCHeadTransform
from common.nlp.roberta import RobertaTokenizer

BERT_WEIGHTS_NAME = 'pytorch_model.bin'


class ResNetVLBERT(Module):
    def __init__(self, config):

        super(ResNetVLBERT, self).__init__(config)

        if not config.NETWORK.BLIND:
            self.image_feature_extractor = FastRCNN(config,
                                                    average_pool=True,
                                                    final_dim=config.NETWORK.IMAGE_FINAL_DIM,
                                                    enable_cnn_reg_loss=(self.enable_cnn_reg_loss and not self.cnn_loss_top))
            if config.NETWORK.VLBERT.object_word_embed_mode == 1:
                self.object_linguistic_embeddings = nn.Embedding(81, config.NETWORK.VLBERT.hidden_size)
            elif config.NETWORK.VLBERT.object_word_embed_mode == 2:
                self.object_linguistic_embeddings = nn.Embedding(1, config.NETWORK.VLBERT.hidden_size)
            elif config.NETWORK.VLBERT.object_word_embed_mode == 3:
                self.object_linguistic_embeddings = None
            else:
                raise NotImplementedError

        self.image_feature_bn_eval = config.NETWORK.IMAGE_FROZEN_BN

        if 'roberta' in config.NETWORK.BERT_MODEL_NAME:
            self.tokenizer = RobertaTokenizer.from_pretrained(config.NETWORK.BERT_MODEL_NAME)
        else:
            self.tokenizer = BertTokenizer.from_pretrained(config.NETWORK.BERT_MODEL_NAME)

        language_pretrained_model_path = None
        if config.NETWORK.BERT_PRETRAINED != '':
            language_pretrained_model_path = '{}-{:04d}.model'.format(config.NETWORK.BERT_PRETRAINED,
                                                                      config.NETWORK.BERT_PRETRAINED_EPOCH)
        elif os.path.isdir(config.NETWORK.BERT_MODEL_NAME):
            weight_path = os.path.join(config.NETWORK.BERT_MODEL_NAME, BERT_WEIGHTS_NAME)
            if os.path.isfile(weight_path):
                language_pretrained_model_path = weight_path

        if language_pretrained_model_path is None:
            print("Warning: no pretrained language model found, training from scratch!!!")

        self.vlbert = TimeDistributed(
            VisualLinguisticBert(config.NETWORK.VLBERT,
                                 language_pretrained_model_path=language_pretrained_model_path)
        )

        dim = config.NETWORK.VLBERT.hidden_size
        if config.NETWORK.CLASSIFIER_TYPE == "2fc":
            self.final_mlp = torch.nn.Sequential(
                torch.nn.Dropout(config.NETWORK.CLASSIFIER_DROPOUT, inplace=False),
                torch.nn.Linear(4*dim, config.NETWORK.CLASSIFIER_HIDDEN_SIZE),
                torch.nn.ReLU(inplace=True),
                torch.nn.Dropout(config.NETWORK.CLASSIFIER_DROPOUT, inplace=False),
                torch.nn.Linear(config.NETWORK.CLASSIFIER_HIDDEN_SIZE, 5),
            )
        elif config.NETWORK.CLASSIFIER_TYPE == "1fc":
            self.final_mlp = torch.nn.Sequential(
                torch.nn.Dropout(config.NETWORK.CLASSIFIER_DROPOUT, inplace=False),
                torch.nn.Linear(4*dim, 5)
            )
        else:
            raise ValueError("Not support classifier type: {}!".format(config.NETWORK.CLASSIFIER_TYPE))

        # init weights
        self.init_weight()

        self.fix_params()

    def init_weight(self):
        if not self.config.NETWORK.BLIND:
            self.image_feature_extractor.init_weight()
            if self.object_linguistic_embeddings is not None:
                self.object_linguistic_embeddings.weight.data.normal_(mean=0.0, std=0.02)

        if not self.for_pretrain:
            for m in self.final_mlp.modules():
                if isinstance(m, torch.nn.Linear):
                    torch.nn.init.xavier_uniform_(m.weight)
                    torch.nn.init.constant_(m.bias, 0)

    def train(self, mode=True):
        super(ResNetVLBERT, self).train(mode)
        # turn some frozen layers to eval mode
        if (not self.config.NETWORK.BLIND) and self.image_feature_bn_eval:
            self.image_feature_extractor.bn_eval()

    def fix_params(self):
        if self.config.NETWORK.BLIND:
            self.vlbert._module.visual_scale_text.requires_grad = False
            self.vlbert._module.visual_scale_object.requires_grad = False

    def _collect_obj_reps(self, span_tags, object_reps):
        """
        Collect span-level object representations
        :param span_tags: [batch_size, ..leading_dims.., L]
        :param object_reps: [batch_size, max_num_objs_per_batch, obj_dim]
        :return:
        """

        span_tags_fixed = torch.clamp(span_tags, min=0)  # In case there were masked values here
        row_id = span_tags_fixed.new_zeros(span_tags_fixed.shape)
        row_id_broadcaster = torch.arange(0, row_id.shape[0], step=1, device=row_id.device)[:, None]

        # Add extra diminsions to the row broadcaster so it matches row_id
        leading_dims = len(span_tags.shape) - 2
        for i in range(leading_dims):
            row_id_broadcaster = row_id_broadcaster[..., None]
        row_id += row_id_broadcaster
        return object_reps[row_id.view(-1), span_tags_fixed.view(-1)].view(*span_tags_fixed.shape, -1)

    def prepare_text(self, sentence1, sentence2, mask1, mask2, phrase1_mask, phrase2_mask):
        batch_size, max_len1 = sentence1.shape
        _, max_len2 = sentence2.shape
        max_len = (mask1.sum(1) + mask2.sum(2).max(1)[0]).max() + 3
        cls_id, sep_id = self.tokenizer.convert_tokens_to_ids(['[CLS]', '[SEP]'])
        end_1 = 1 + mask1.sum(2, keepdim=True)
        end_2 = end_1 + 1 + mask2.sum(2, keepdim=True)
        input_ids = torch.zeros((batch_size, max_len), dtype=sentence1.dtype, device=sentence1.device)
        input_mask = torch.ones((batch_size, max_len), dtype=torch.uint8, device=sentence1.device)
        input_type_ids = torch.zeros((batch_size, max_len), dtype=sentence1.dtype, device=sentence1.device)
        text_tags = input_type_ids.new_zeros((batch_size, max_len))
        grid_i, grid_k = torch.meshgrid(torch.arange(batch_size, device=sentence1.device),
                                        torch.arange(max_len, device=sentence1.device))

        input_mask[grid_k > end_2] = 0
        input_type_ids[(grid_k > end_1) & (grid_k <= end_2)] = 1
        input_mask1 = (grid_k > 0) & (grid_k < end_1)
        input_mask2 = (grid_k > end_1) & (grid_k < end_2)
        input_ids[:, :, 0] = cls_id
        input_ids[grid_k == end_1] = sep_id
        input_ids[grid_k == end_2] = sep_id
        input_ids[input_mask1] = sentence1[mask1]
        input_ids[input_mask2] = sentence2[mask2]
        # text_tags[q_input_mask] = question_tags[question_mask]
        # text_tags[a_input_mask] = answers_tags[answers_mask]
        ph1_mask = torch.zeros((batch_size, max_len), dtype=sentence1.dtype, device=sentence1.device)
        ph1_mask[input_mask1] = phrase1_mask
        ph2_mask = torch.zeros((batch_size, max_len), dtype=sentence1.dtype, device=sentence1.device)
        ph2_mask[input_mask1] = phrase2_mask

        return input_ids, input_type_ids, text_tags, input_mask, ph1_mask, ph2_mask

    def train_forward(self,
                      images,
                      boxes,
                      masks,
                      sentence1,
                      sentence2,
                      im_info,
                      paraphrase_label):
        ###########################################
        # visual feature extraction

        # Don't know what segments are for
        # segms = masks

        # For now use all boxes
        box_mask = torch.ones(boxes[:, :, -1].size(), dtype=torch.uint8)

        max_len = int(box_mask.sum(1).max().item())
        box_mask = box_mask[:, :max_len]
        boxes = boxes[:, :max_len]

        # segms = segms[:, :max_len]

        if self.config.NETWORK.BLIND:
            obj_reps = {'obj_reps': boxes.new_zeros((*boxes.shape[:-1], self.config.NETWORK.IMAGE_FINAL_DIM))}
        else:
            obj_reps = self.image_feature_extractor(images=images,
                                                    boxes=boxes,
                                                    box_mask=box_mask,
                                                    im_info=im_info,
                                                    classes=None,
                                                    segms=None)

        # For now no tags
        sentence1_ids = sentence1[:, :, 0]
        mask1 = (sentence1[:, :, 0] > 0.5)
        phrase1_mask = sentence1[:, :, 1]
        sentence2_ids = sentence2[:, :, :0]
        mask2 = (sentence2[:, :, 0] > 0.5)
        phrase2_mask = sentence2[:, :, 1]

        ############################################

        # prepare text
        text_input_ids, text_token_type_ids, text_tags, text_mask, ph1_mask, ph2_mask = self.prepare_text(sentence1_ids,
                                                                                                          sentence2_ids,
                                                                                                          mask1,
                                                                                                          mask2,
                                                                                                          phrase1_mask,
                                                                                                          phrase2_mask)

        # Add visual feature to text elements
        if self.config.NETWORK.NO_GROUNDING:
            text_tags.zero_()
        text_visual_embeddings = self._collect_obj_reps(text_tags, obj_reps['obj_reps'])

        # Add textual feature to image element
        if self.config.NETWORK.BLIND:
            object_linguistic_embeddings = boxes.new_zeros((*boxes.shape[:-1], self.config.NETWORK.VLBERT.hidden_size))
        else:
            object_linguistic_embeddings = self.object_linguistic_embeddings(
                boxes.new_zeros((boxes.shape[0], boxes.shape[1])).long())
        object_vl_embeddings = torch.cat((obj_reps['obj_reps'], object_linguistic_embeddings), -1)

        ###########################################

        # Visual Linguistic BERT

        if self.config.NETWORK.NO_OBJ_ATTENTION or self.config.NETWORK.BLIND:
            box_mask.zero_()

        hidden_states_text, hidden_states_objects, pooled_rep = self.vlbert(text_input_ids,
                                                                            text_token_type_ids,
                                                                            text_visual_embeddings,
                                                                            text_mask,
                                                                            object_vl_embeddings,
                                                                            box_mask,
                                                                            output_all_encoded_layers=False,
                                                                            output_text_and_object_separately=True)

        ###########################################
        outputs = {}

        # max pooling representation of each phrase
        h_rep_ph1 = hidden_states_text[:, ph1_mask, :].max(axis=1)
        h_rep_ph2 = hidden_states_text[:, ph2_mask, :].max(axis=1)
        final_rep = torch.cat((h_rep_ph1, h_rep_ph2, torch.abs(h_rep_ph2 - h_rep_ph1), torch.mul(h_rep_ph1, h_rep_ph2)),
                              axis=1)

        # classifier
        logits = self.final_mlp(final_rep)

        # loss
        cls_loss = F.cross_entropy(logits, paraphrase_label.long().view(-1))

        outputs.update({'label_logits': logits,
                        'label': paraphrase_label.long().view(-1),
                        'cls_loss': cls_loss})

        loss = cls_loss.mean() * self.config.NETWORK.ANS_LOSS_WEIGHT

        return outputs, loss

    def inference_forward(self,
                          image,
                          boxes,
                          masks,
                          question,
                          question_align_matrix,
                          answer_choices,
                          answer_align_matrix,
                          *args):

        if self.for_pretrain:
            answer_label, im_info, mask_position, mask_type = args
        else:
            assert len(args) == 1
            im_info = args[0]

        ###########################################

        # visual feature extraction
        images = image
        objects = boxes[:, :, -1]
        segms = masks
        boxes = boxes[:, :, :4]
        box_mask = (boxes[:, :, -1] > - 0.5)
        max_len = int(box_mask.sum(1).max().item())
        objects = objects[:, :max_len]
        box_mask = box_mask[:, :max_len]
        boxes = boxes[:, :max_len]
        segms = segms[:, :max_len]

        if self.config.NETWORK.BLIND:
            obj_reps = {'obj_reps': boxes.new_zeros((*boxes.shape[:-1], self.config.NETWORK.IMAGE_FINAL_DIM))}
        else:
            obj_reps = self.image_feature_extractor(images=images,
                                                    boxes=boxes,
                                                    box_mask=box_mask,
                                                    im_info=im_info,
                                                    classes=objects,
                                                    segms=segms)

        num_choices = answer_choices.shape[1]
        question_ids = question[:, :, 0]
        question_tags = question[:, :, 1]
        question_tags = question_tags.repeat(1, num_choices).view(question_tags.shape[0], num_choices, -1)
        question_mask = (question[:, :, 0] > 0.5)
        answer_ids = answer_choices[:, :, :, 0]
        answer_tags = answer_choices[:, :, :, 1]
        answer_mask = (answer_choices[:, :, :, 0] > 0.5)

        ############################################

        # prepare text
        text_input_ids, text_token_type_ids, text_tags, text_mask = self.prepare_text_from_qa(
            question_ids,
            question_tags,
            question_mask,
            answer_ids,
            answer_tags,
            answer_mask)

        if self.config.NETWORK.NO_GROUNDING:
            text_tags.zero_()
        text_visual_embeddings = self._collect_obj_reps(text_tags, obj_reps['obj_reps'])
        if self.config.NETWORK.BLIND:
            object_linguistic_embeddings = boxes.new_zeros(
                (*boxes.shape[:-1], self.config.NETWORK.VLBERT.hidden_size))
            object_linguistic_embeddings = object_linguistic_embeddings.unsqueeze(1).repeat(1, num_choices, 1, 1)
        else:
            if self.config.NETWORK.VLBERT.object_word_embed_mode in [1, 2]:
                object_linguistic_embeddings = self.object_linguistic_embeddings(
                    objects.long().clamp(min=0, max=self.object_linguistic_embeddings.weight.data.shape[0] - 1)
                )
                object_linguistic_embeddings = object_linguistic_embeddings.unsqueeze(1).repeat(1, num_choices, 1,
                                                                                                1)
            elif self.config.NETWORK.VLBERT.object_word_embed_mode == 3:
                cls_id, sep_id = self.tokenizer.convert_tokens_to_ids(['[CLS]', '[SEP]'])
                global_context_mask = text_mask & (text_input_ids != cls_id) & (text_input_ids != sep_id)
                word_embedding = self.vlbert._module.word_embeddings(text_input_ids)
                word_embedding[global_context_mask == 0] = 0
                object_linguistic_embeddings = word_embedding.sum(dim=2) / global_context_mask.sum(dim=2,
                                                                                                   keepdim=True).to(
                    dtype=word_embedding.dtype)
                object_linguistic_embeddings = object_linguistic_embeddings.unsqueeze(2).repeat((1, 1, max_len, 1))
        object_vl_embeddings = torch.cat((obj_reps['obj_reps'].unsqueeze(1).repeat(1, num_choices, 1, 1),
                                          object_linguistic_embeddings), -1)

        ###########################################

        # Visual Linguistic BERT

        if self.config.NETWORK.NO_OBJ_ATTENTION or self.config.NETWORK.BLIND:
            box_mask.zero_()

        hidden_states_text, hidden_states_objects, pooled_rep = self.vlbert(text_input_ids,
                                                                          text_token_type_ids,
                                                                          text_visual_embeddings,
                                                                          text_mask,
                                                                          object_vl_embeddings,
                                                                          box_mask.unsqueeze(1).repeat(1,
                                                                                                       num_choices,
                                                                                                       1),
                                                                          output_all_encoded_layers=False,
                                                                          output_text_and_object_separately=True)

        ###########################################

        # classifier
        logits = self.final_mlp(pooled_rep).squeeze(2)

        outputs = {'label_logits': logits}

        return outputs

