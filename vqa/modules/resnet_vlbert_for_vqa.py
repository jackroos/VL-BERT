import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from external.pytorch_pretrained_bert import BertTokenizer
from external.pytorch_pretrained_bert.modeling import BertPredictionHeadTransform
from common.module import Module
from common.fast_rcnn import FastRCNN
from common.visual_linguistic_bert import VisualLinguisticBert

BERT_WEIGHTS_NAME = 'pytorch_model.bin'


class ResNetVLBERT(Module):
    def __init__(self, config):

        super(ResNetVLBERT, self).__init__(config)

        self.enable_cnn_reg_loss = config.NETWORK.ENABLE_CNN_REG_LOSS
        if not config.NETWORK.BLIND:
            self.image_feature_extractor = FastRCNN(config,
                                                    average_pool=True,
                                                    final_dim=config.NETWORK.IMAGE_FINAL_DIM,
                                                    enable_cnn_reg_loss=self.enable_cnn_reg_loss)
            if config.NETWORK.VLBERT.object_word_embed_mode == 1:
                self.object_linguistic_embeddings = nn.Embedding(81, config.NETWORK.VLBERT.hidden_size)
            elif config.NETWORK.VLBERT.object_word_embed_mode == 2:
                self.object_linguistic_embeddings = nn.Embedding(1, config.NETWORK.VLBERT.hidden_size)
            elif config.NETWORK.VLBERT.object_word_embed_mode == 3:
                self.object_linguistic_embeddings = None
            else:
                raise NotImplementedError
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

        # self.hm_out = nn.Linear(config.NETWORK.VLBERT.hidden_size, config.NETWORK.VLBERT.hidden_size)
        # self.hi_out = nn.Linear(config.NETWORK.VLBERT.hidden_size, config.NETWORK.VLBERT.hidden_size)

        dim = config.NETWORK.VLBERT.hidden_size
        if config.NETWORK.CLASSIFIER_TYPE == "2fc":
            self.final_mlp = torch.nn.Sequential(
                torch.nn.Dropout(config.NETWORK.CLASSIFIER_DROPOUT, inplace=False),
                torch.nn.Linear(dim, config.NETWORK.CLASSIFIER_HIDDEN_SIZE),
                torch.nn.ReLU(inplace=True),
                torch.nn.Dropout(config.NETWORK.CLASSIFIER_DROPOUT, inplace=False),
                torch.nn.Linear(config.NETWORK.CLASSIFIER_HIDDEN_SIZE, config.DATASET.ANSWER_VOCAB_SIZE),
            )
        elif config.NETWORK.CLASSIFIER_TYPE == "1fc":
            self.final_mlp = torch.nn.Sequential(
                torch.nn.Dropout(config.NETWORK.CLASSIFIER_DROPOUT, inplace=False),
                torch.nn.Linear(dim, config.DATASET.ANSWER_VOCAB_SIZE)
            )
        elif config.NETWORK.CLASSIFIER_TYPE == 'mlm':
            transform = BertPredictionHeadTransform(config.NETWORK.VLBERT)
            linear = nn.Linear(config.NETWORK.VLBERT.hidden_size, config.DATASET.ANSWER_VOCAB_SIZE)
            self.final_mlp = nn.Sequential(
                transform,
                nn.Dropout(config.NETWORK.CLASSIFIER_DROPOUT, inplace=False),
                linear
            )
        else:
            raise ValueError("Not support classifier type: {}!".format(config.NETWORK.CLASSIFIER_TYPE))

        # init weights
        self.init_weight()

        self.fix_params()

    def init_weight(self):
        # self.hm_out.weight.data.normal_(mean=0.0, std=0.02)
        # self.hm_out.bias.data.zero_()
        # self.hi_out.weight.data.normal_(mean=0.0, std=0.02)
        # self.hi_out.bias.data.zero_()
        self.image_feature_extractor.init_weight()
        if self.object_linguistic_embeddings is not None:
            self.object_linguistic_embeddings.weight.data.normal_(mean=0.0, std=0.02)
        for m in self.final_mlp.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.constant_(m.bias, 0)
        if self.config.NETWORK.CLASSIFIER_TYPE == 'mlm':
            language_pretrained = torch.load(self.language_pretrained_model_path)
            mlm_transform_state_dict = {}
            pretrain_keys = []
            for k, v in language_pretrained.items():
                if k.startswith('cls.predictions.transform.'):
                    pretrain_keys.append(k)
                    k_ = k[len('cls.predictions.transform.'):]
                    if 'gamma' in k_:
                        k_ = k_.replace('gamma', 'weight')
                    if 'beta' in k_:
                        k_ = k_.replace('beta', 'bias')
                    mlm_transform_state_dict[k_] = v
            print("loading pretrained classifier transform keys: {}.".format(pretrain_keys))
            self.final_mlp[0].load_state_dict(mlm_transform_state_dict)

    def train(self, mode=True):
        super(ResNetVLBERT, self).train(mode)
        # turn some frozen layers to eval mode
        if self.image_feature_bn_eval:
            self.image_feature_extractor.bn_eval()

    def fix_params(self):
        pass

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

    def prepare_text_from_qa(self, question, question_tags, question_mask, answer, answer_tags, answer_mask):
        batch_size, max_q_len = question.shape
        _, max_a_len = answer.shape
        max_len = (question_mask.sum(1) + answer_mask.sum(1)).max() + 3
        cls_id, sep_id = self.tokenizer.convert_tokens_to_ids(['[CLS]', '[SEP]'])
        q_end = 1 + question_mask.sum(1, keepdim=True)
        a_end = q_end + 1 + answer_mask.sum(1, keepdim=True)
        input_ids = torch.zeros((batch_size, max_len), dtype=question.dtype, device=question.device)
        input_mask = torch.ones((batch_size, max_len), dtype=torch.uint8, device=question.device)
        input_type_ids = torch.zeros((batch_size, max_len), dtype=question.dtype, device=question.device)
        text_tags = input_type_ids.new_zeros((batch_size, max_len))
        grid_i, grid_j = torch.meshgrid(torch.arange(batch_size, device=question.device),
                                        torch.arange(max_len, device=question.device))

        input_mask[grid_j > a_end] = 0
        input_type_ids[(grid_j > q_end) & (grid_j <= a_end)] = 1
        q_input_mask = (grid_j > 0) & (grid_j < q_end)
        a_input_mask = (grid_j > q_end) & (grid_j < a_end)
        input_ids[:, 0] = cls_id
        input_ids[grid_j == q_end] = sep_id
        input_ids[grid_j == a_end] = sep_id
        input_ids[q_input_mask] = question[question_mask]
        input_ids[a_input_mask] = answer[answer_mask]
        text_tags[q_input_mask] = question_tags[question_mask]
        text_tags[a_input_mask] = answer_tags[answer_mask]

        return input_ids, input_type_ids, text_tags, input_mask, (a_end - 1).squeeze(1)

    def train_forward(self,
                      image,
                      boxes,
                      im_info,
                      question,
                      label,
                      ):
        ###########################################

        # visual feature extraction
        images = image
        box_mask = (boxes[:, :, 0] > - 1.5)
        max_len = int(box_mask.sum(1).max().item())
        box_mask = box_mask[:, :max_len]
        boxes = boxes[:, :max_len]

        obj_reps = self.image_feature_extractor(images=images,
                                                boxes=boxes,
                                                box_mask=box_mask,
                                                im_info=im_info,
                                                classes=None,
                                                segms=None)

        question_ids = question
        question_tags = question.new_zeros(question_ids.shape)
        question_mask = (question > 0.5)

        answer_ids = question_ids.new_zeros((question_ids.shape[0], 1)).fill_(
            self.tokenizer.convert_tokens_to_ids(['[MASK]'])[0])
        answer_mask = question_mask.new_zeros(answer_ids.shape).fill_(1)
        answer_tags = question_tags.new_zeros(answer_ids.shape)

        ############################################

        # prepare text
        text_input_ids, text_token_type_ids, text_tags, text_mask, ans_pos = self.prepare_text_from_qa(question_ids,
                                                                                                       question_tags,
                                                                                                       question_mask,
                                                                                                       answer_ids,
                                                                                                       answer_tags,
                                                                                                       answer_mask)
        if self.config.NETWORK.NO_GROUNDING:
            obj_rep_zeroed = obj_reps['obj_reps'].new_zeros(obj_reps['obj_reps'].shape)
            text_tags.zero_()
            text_visual_embeddings = self._collect_obj_reps(text_tags, obj_rep_zeroed)
        else:
            text_visual_embeddings = self._collect_obj_reps(text_tags, obj_reps['obj_reps'])

        assert self.config.NETWORK.VLBERT.object_word_embed_mode == 2
        object_linguistic_embeddings = self.object_linguistic_embeddings(
            boxes.new_zeros((boxes.shape[0], boxes.shape[1])).long()
        )
        object_vl_embeddings = torch.cat((obj_reps['obj_reps'], object_linguistic_embeddings), -1)

        ###########################################

        # Visual Linguistic BERT

        hidden_states, hc = self.vlbert(text_input_ids,
                                      text_token_type_ids,
                                      text_visual_embeddings,
                                      text_mask,
                                      object_vl_embeddings,
                                      box_mask,
                                      output_all_encoded_layers=False)
        _batch_inds = torch.arange(question.shape[0], device=question.device)

        hm = hidden_states[_batch_inds, ans_pos]
        # hm = F.tanh(self.hm_out(hidden_states[_batch_inds, ans_pos]))
        # hi = F.tanh(self.hi_out(hidden_states[_batch_inds, ans_pos + 2]))

        ###########################################
        outputs = {}

        # classifier
        # logits = self.final_mlp(hc * hm * hi)
        # logits = self.final_mlp(hc)
        logits = self.final_mlp(hm)

        # loss
        ans_loss = F.binary_cross_entropy_with_logits(logits, label) * label.size(1)

        outputs.update({'label_logits': logits,
                        'label': label,
                        'ans_loss': ans_loss})

        loss = ans_loss.mean()

        return outputs, loss

    def inference_forward(self,
                          image,
                          boxes,
                          im_info,
                          question):

        ###########################################

        # visual feature extraction
        images = image
        box_mask = (boxes[:, :, 0] > - 1.5)
        max_len = int(box_mask.sum(1).max().item())
        box_mask = box_mask[:, :max_len]
        boxes = boxes[:, :max_len]

        obj_reps = self.image_feature_extractor(images=images,
                                                boxes=boxes,
                                                box_mask=box_mask,
                                                im_info=im_info,
                                                classes=None,
                                                segms=None)

        question_ids = question
        question_tags = question.new_zeros(question_ids.shape)
        question_mask = (question > 0.5)

        answer_ids = question_ids.new_zeros((question_ids.shape[0], 1)).fill_(
            self.tokenizer.convert_tokens_to_ids(['[MASK]'])[0])
        answer_mask = question_mask.new_zeros(answer_ids.shape).fill_(1)
        answer_tags = question_tags.new_zeros(answer_ids.shape)

        ############################################

        # prepare text
        text_input_ids, text_token_type_ids, text_tags, text_mask, ans_pos = self.prepare_text_from_qa(question_ids,
                                                                                                       question_tags,
                                                                                                       question_mask,
                                                                                                       answer_ids,
                                                                                                       answer_tags,
                                                                                                       answer_mask)
        if self.config.NETWORK.NO_GROUNDING:
            obj_rep_zeroed = obj_reps['obj_reps'].new_zeros(obj_reps['obj_reps'].shape)
            text_tags.zero_()
            text_visual_embeddings = self._collect_obj_reps(text_tags, obj_rep_zeroed)
        else:
            text_visual_embeddings = self._collect_obj_reps(text_tags, obj_reps['obj_reps'])

        assert self.config.NETWORK.VLBERT.object_word_embed_mode == 2
        object_linguistic_embeddings = self.object_linguistic_embeddings(
            boxes.new_zeros((boxes.shape[0], boxes.shape[1])).long()
        )
        object_vl_embeddings = torch.cat((obj_reps['obj_reps'], object_linguistic_embeddings), -1)

        ###########################################

        # Visual Linguistic BERT

        hidden_states, hc = self.vlbert(text_input_ids,
                                      text_token_type_ids,
                                      text_visual_embeddings,
                                      text_mask,
                                      object_vl_embeddings,
                                      box_mask,
                                      output_all_encoded_layers=False)
        _batch_inds = torch.arange(question.shape[0], device=question.device)

        hm = hidden_states[_batch_inds, ans_pos]
        # hm = F.tanh(self.hm_out(hidden_states[_batch_inds, ans_pos]))
        # hi = F.tanh(self.hi_out(hidden_states[_batch_inds, ans_pos + 2]))

        ###########################################
        outputs = {}

        # classifier
        # logits = self.final_mlp(hc * hm * hi)
        # logits = self.final_mlp(hc)
        logits = self.final_mlp(hm)

        outputs.update({'label_logits': logits})

        return outputs
