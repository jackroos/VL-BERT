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

        self.enable_cnn_reg_loss = config.NETWORK.ENABLE_CNN_REG_LOSS
        self.cnn_loss_top = config.NETWORK.CNN_LOSS_TOP
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
            if self.enable_cnn_reg_loss and self.cnn_loss_top:
                self.cnn_loss_reg = nn.Sequential(
                    VisualLinguisticBertMVRCHeadTransform(config.NETWORK.VLBERT),
                    nn.Dropout(config.NETWORK.CNN_REG_DROPOUT, inplace=False),
                    nn.Linear(config.NETWORK.VLBERT.hidden_size, 81)
                )
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

        self.for_pretrain = config.NETWORK.FOR_MASK_VL_MODELING_PRETRAIN
        assert not self.for_pretrain, "Not implement pretrain mode now!"

        if not self.for_pretrain:
            dim = config.NETWORK.VLBERT.hidden_size
            if config.NETWORK.CLASSIFIER_TYPE == "2fc":
                self.final_mlp = torch.nn.Sequential(
                    torch.nn.Dropout(config.NETWORK.CLASSIFIER_DROPOUT, inplace=False),
                    torch.nn.Linear(dim, config.NETWORK.CLASSIFIER_HIDDEN_SIZE),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Dropout(config.NETWORK.CLASSIFIER_DROPOUT, inplace=False),
                    torch.nn.Linear(config.NETWORK.CLASSIFIER_HIDDEN_SIZE, 1),
                )
            elif config.NETWORK.CLASSIFIER_TYPE == "1fc":
                self.final_mlp = torch.nn.Sequential(
                    torch.nn.Dropout(config.NETWORK.CLASSIFIER_DROPOUT, inplace=False),
                    torch.nn.Linear(dim, 1)
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
            if self.enable_cnn_reg_loss and self.cnn_loss_top:
                self.cnn_loss_reg.apply(self.vlbert._module.init_weights)

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

    def prepare_text_from_qa(self, question, question_tags, question_mask, answers, answers_tags, answers_mask):
        batch_size, max_q_len = question.shape
        _, num_choices, max_a_len = answers.shape
        max_len = (question_mask.sum(1) + answers_mask.sum(2).max(1)[0]).max() + 3
        cls_id, sep_id = self.tokenizer.convert_tokens_to_ids(['[CLS]', '[SEP]'])
        question = question.repeat(1, num_choices).view(-1, num_choices, max_q_len)
        question_mask = question_mask.repeat(1, num_choices).view(-1, num_choices, max_q_len)
        q_end = 1 + question_mask.sum(2, keepdim=True)
        a_end = q_end + 1 + answers_mask.sum(2, keepdim=True)
        input_ids = torch.zeros((batch_size, num_choices, max_len), dtype=question.dtype, device=question.device)
        input_mask = torch.ones((batch_size, num_choices, max_len), dtype=torch.uint8, device=question.device)
        input_type_ids = torch.zeros((batch_size, num_choices, max_len), dtype=question.dtype, device=question.device)
        text_tags = input_type_ids.new_zeros((batch_size, num_choices, max_len))
        grid_i, grid_j, grid_k = torch.meshgrid(torch.arange(batch_size, device=question.device),
                                                torch.arange(num_choices, device=question.device),
                                                torch.arange(max_len, device=question.device))

        input_mask[grid_k > a_end] = 0
        input_type_ids[(grid_k > q_end) & (grid_k <= a_end)] = 1
        q_input_mask = (grid_k > 0) & (grid_k < q_end)
        a_input_mask = (grid_k > q_end) & (grid_k < a_end)
        input_ids[:, :, 0] = cls_id
        input_ids[grid_k == q_end] = sep_id
        input_ids[grid_k == a_end] = sep_id
        input_ids[q_input_mask] = question[question_mask]
        input_ids[a_input_mask] = answers[answers_mask]
        text_tags[q_input_mask] = question_tags[question_mask]
        text_tags[a_input_mask] = answers_tags[answers_mask]

        return input_ids, input_type_ids, text_tags, input_mask

    def prepare_text_from_qa_onesent(self, question, question_tags, question_mask, answers, answers_tags, answers_mask):
        batch_size, max_q_len = question.shape
        _, num_choices, max_a_len = answers.shape
        max_len = (question_mask.sum(1) + answers_mask.sum(2).max(1)[0]).max() + 2
        cls_id, sep_id = self.tokenizer.convert_tokens_to_ids(['[CLS]', '[SEP]'])
        question = question.repeat(1, num_choices).view(-1, num_choices, max_q_len)
        question_mask = question_mask.repeat(1, num_choices).view(-1, num_choices, max_q_len)
        q_end = 1 + question_mask.sum(2, keepdim=True)
        a_end = q_end + answers_mask.sum(2, keepdim=True)
        input_ids = torch.zeros((batch_size, num_choices, max_len), dtype=question.dtype, device=question.device)
        input_mask = torch.ones((batch_size, num_choices, max_len), dtype=torch.uint8, device=question.device)
        input_type_ids = torch.zeros((batch_size, num_choices, max_len), dtype=question.dtype, device=question.device)
        text_tags = input_type_ids.new_zeros((batch_size, num_choices, max_len))
        grid_i, grid_j, grid_k = torch.meshgrid(torch.arange(batch_size, device=question.device),
                                                torch.arange(num_choices, device=question.device),
                                                torch.arange(max_len, device=question.device))

        input_mask[grid_k > a_end] = 0
        q_input_mask = (grid_k > 0) & (grid_k < q_end)
        a_input_mask = (grid_k >= q_end) & (grid_k < a_end)
        input_ids[:, :, 0] = cls_id
        input_ids[grid_k == a_end] = sep_id
        input_ids[q_input_mask] = question[question_mask]
        input_ids[a_input_mask] = answers[answers_mask]
        text_tags[q_input_mask] = question_tags[question_mask]
        text_tags[a_input_mask] = answers_tags[answers_mask]

        return input_ids, input_type_ids, text_tags, input_mask

    def prepare_text_from_aq(self, question, question_tags, question_mask, answers, answers_tags, answers_mask):
        batch_size, max_q_len = question.shape
        _, num_choices, max_a_len = answers.shape
        max_len = (question_mask.sum(1) + answers_mask.sum(2).max(1)[0]).max() + 3
        cls_id, sep_id = self.tokenizer.convert_tokens_to_ids(['[CLS]', '[SEP]'])
        question = question.repeat(1, num_choices).view(-1, num_choices, max_q_len)
        question_mask = question_mask.repeat(1, num_choices).view(-1, num_choices, max_q_len)
        a_end = 1 + answers_mask.sum(2, keepdim=True)
        q_end = a_end + 1 + question_mask.sum(2, keepdim=True)
        input_ids = torch.zeros((batch_size, num_choices, max_len), dtype=question.dtype, device=question.device)
        input_mask = torch.ones((batch_size, num_choices, max_len), dtype=torch.uint8, device=question.device)
        input_type_ids = torch.zeros((batch_size, num_choices, max_len), dtype=question.dtype, device=question.device)
        text_tags = input_type_ids.new_zeros((batch_size, num_choices, max_len))
        grid_i, grid_j, grid_k = torch.meshgrid(torch.arange(batch_size, device=question.device),
                                                torch.arange(num_choices, device=question.device),
                                                torch.arange(max_len, device=question.device))

        input_mask[grid_k > q_end] = 0
        input_type_ids[(grid_k > a_end) & (grid_k <= q_end)] = 1
        q_input_mask = (grid_k > a_end) & (grid_k < q_end)
        a_input_mask = (grid_k > 0) & (grid_k < a_end)
        input_ids[:, :, 0] = cls_id
        input_ids[grid_k == a_end] = sep_id
        input_ids[grid_k == q_end] = sep_id
        input_ids[q_input_mask] = question[question_mask]
        input_ids[a_input_mask] = answers[answers_mask]
        text_tags[q_input_mask] = question_tags[question_mask]
        text_tags[a_input_mask] = answers_tags[answers_mask]

        return input_ids, input_type_ids, text_tags, input_mask

    def train_forward(self,
                      image,
                      boxes,
                      masks,
                      question,
                      question_align_matrix,
                      answer_choices,
                      answer_align_matrix,
                      answer_label,
                      im_info,
                      mask_position=None,
                      mask_type=None,
                      mask_label=None):
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
        answer_ids = answer_choices[:, :, :,0]
        answer_tags = answer_choices[:, :, :, 1]
        answer_mask = (answer_choices[:, :, :, 0] > 0.5)

        ############################################

        # prepare text
        if self.config.NETWORK.ANSWER_FIRST:
            if self.config.NETWORK.QA_ONE_SENT:
                raise NotImplemented
            else:
                text_input_ids, text_token_type_ids, text_tags, text_mask = self.prepare_text_from_aq(
                    question_ids,
                    question_tags,
                    question_mask,
                    answer_ids,
                    answer_tags,
                    answer_mask)
        else:
            if self.config.NETWORK.QA_ONE_SENT:
                text_input_ids, text_token_type_ids, text_tags, text_mask = self.prepare_text_from_qa_onesent(
                    question_ids,
                    question_tags,
                    question_mask,
                    answer_ids,
                    answer_tags,
                    answer_mask)
            else:
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
            object_linguistic_embeddings = boxes.new_zeros((*boxes.shape[:-1], self.config.NETWORK.VLBERT.hidden_size))
            object_linguistic_embeddings = object_linguistic_embeddings.unsqueeze(1).repeat(1, num_choices, 1, 1)
        else:
            if self.config.NETWORK.VLBERT.object_word_embed_mode in [1, 2]:
                object_linguistic_embeddings = self.object_linguistic_embeddings(
                    objects.long().clamp(min=0, max=self.object_linguistic_embeddings.weight.data.shape[0] - 1)
                )
                object_linguistic_embeddings = object_linguistic_embeddings.unsqueeze(1).repeat(1, num_choices, 1, 1)
            elif self.config.NETWORK.VLBERT.object_word_embed_mode == 3:
                cls_id, sep_id = self.tokenizer.convert_tokens_to_ids(['[CLS]', '[SEP]'])
                global_context_mask = text_mask & (text_input_ids != cls_id) & (text_input_ids != sep_id)
                word_embedding = self.vlbert._module.word_embeddings(text_input_ids)
                word_embedding[global_context_mask == 0] = 0
                object_linguistic_embeddings = word_embedding.sum(dim=2) / global_context_mask.sum(dim=2, keepdim=True).to(dtype=word_embedding.dtype)
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
                                                                          box_mask.unsqueeze(1).repeat(1, num_choices, 1),
                                                                          output_all_encoded_layers=False,
                                                                          output_text_and_object_separately=True)

        ###########################################
        outputs = {}

        # classifier
        logits = self.final_mlp(pooled_rep).squeeze(2)

        # loss
        if self.config.NETWORK.CLASSIFIER_SIGMOID:
            _, choice_ind = torch.meshgrid(torch.arange(logits.shape[0], device=logits.device),
                                           torch.arange(num_choices, device=logits.device))
            label_binary = (choice_ind == answer_label.unsqueeze(1))
            if mask_type is not None and self.config.NETWORK.REPLACE_OBJECT_CHANGE_LABEL:
                label_binary = label_binary * (mask_type != 1).unsqueeze(1)
            weight = logits.new_zeros(logits.shape).fill_(1.0)
            weight[label_binary == 1] = self.config.NETWORK.CLASSIFIER_SIGMOID_LOSS_POSITIVE_WEIGHT
            rescale = (self.config.NETWORK.CLASSIFIER_SIGMOID_LOSS_POSITIVE_WEIGHT + 1.0) \
                / (2.0 * self.config.NETWORK.CLASSIFIER_SIGMOID_LOSS_POSITIVE_WEIGHT)
            ans_loss = rescale * F.binary_cross_entropy_with_logits(logits, label_binary.to(dtype=logits.dtype),
                                                                    weight=weight)
            outputs['positive_fraction'] = label_binary.to(dtype=logits.dtype).sum() / label_binary.numel()
        else:
            ans_loss = F.cross_entropy(logits, answer_label.long().view(-1))

        outputs.update({'label_logits': logits,
                        'label': answer_label.long().view(-1),
                        'ans_loss': ans_loss})

        loss = ans_loss.mean() * self.config.NETWORK.ANS_LOSS_WEIGHT

        if mask_position is not None:
            assert False, "Todo: align to original position."
            _batch_ind = torch.arange(images.shape[0], dtype=torch.long, device=images.device)
            mask_pos_rep = hidden_states[_batch_ind, answer_label, mask_position]
            mask_pred_logits = (obj_reps['obj_reps'] @ mask_pos_rep.unsqueeze(-1)).squeeze(-1)
            mask_pred_logits[1 - box_mask] -= 10000.0
            mask_object_loss = F.cross_entropy(mask_pred_logits, mask_label, ignore_index=-1)
            logits_padded = mask_pred_logits.new_zeros((mask_pred_logits.shape[0], origin_len)).fill_(-10000.0)
            logits_padded[:, :mask_pred_logits.shape[1]] = mask_pred_logits
            mask_pred_logits = logits_padded
            outputs.update({
                'mask_object_loss': mask_object_loss,
                'mask_object_logits': mask_pred_logits,
                'mask_object_label': mask_label})
            loss = loss + mask_object_loss.mean() * self.config.NETWORK.MASK_OBJECT_LOSS_WEIGHT

        if self.enable_cnn_reg_loss:
            if not self.cnn_loss_top:
                loss = loss + obj_reps['cnn_regularization_loss'].mean() * self.config.NETWORK.CNN_LOSS_WEIGHT
                outputs['cnn_regularization_loss'] = obj_reps['cnn_regularization_loss']
            else:
                objects = objects.unsqueeze(1).repeat(1, num_choices, 1)
                box_mask = box_mask.unsqueeze(1).repeat(1, num_choices, 1)
                cnn_reg_logits = self.cnn_loss_reg(hidden_states_objects[box_mask])
                cnn_reg_loss = F.cross_entropy(cnn_reg_logits, objects[box_mask].long())
                loss = loss + cnn_reg_loss.mean() * self.config.NETWORK.CNN_LOSS_WEIGHT
                outputs['cnn_regularization_loss'] = cnn_reg_loss

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
        if self.config.NETWORK.ANSWER_FIRST:
            if self.config.NETWORK.QA_ONE_SENT:
                raise NotImplemented
            else:
                text_input_ids, text_token_type_ids, text_tags, text_mask = self.prepare_text_from_aq(
                    question_ids,
                    question_tags,
                    question_mask,
                    answer_ids,
                    answer_tags,
                    answer_mask)
        else:
            if self.config.NETWORK.QA_ONE_SENT:
                text_input_ids, text_token_type_ids, text_tags, text_mask = self.prepare_text_from_qa_onesent(
                    question_ids,
                    question_tags,
                    question_mask,
                    answer_ids,
                    answer_tags,
                    answer_mask)
            else:
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

