import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from external.pytorch_pretrained_bert import BertTokenizer
from common.module import Module
from common.fast_rcnn import FastRCNN
from common.visual_linguistic_bert import VisualLinguisticBert
from common.utils.misc import soft_cross_entropy

BERT_WEIGHTS_NAME = 'pytorch_model.bin'


class ResNetVLBERTForAttentionVis(Module):
    def __init__(self, config):

        super(ResNetVLBERTForAttentionVis, self).__init__(config)

        self.image_feature_extractor = FastRCNN(config,
                                                average_pool=True,
                                                final_dim=config.NETWORK.IMAGE_FINAL_DIM,
                                                enable_cnn_reg_loss=False)
        self.object_linguistic_embeddings = nn.Embedding(1, config.NETWORK.VLBERT.hidden_size)
        if config.NETWORK.IMAGE_FEAT_PRECOMPUTED or (not config.NETWORK.MASK_RAW_PIXELS):
            self.object_mask_visual_embedding = nn.Embedding(1, 2048)
        if config.NETWORK.WITH_MVRC_LOSS:
            self.object_mask_word_embedding = nn.Embedding(1, config.NETWORK.VLBERT.hidden_size)
        self.aux_text_visual_embedding = nn.Embedding(1, config.NETWORK.VLBERT.hidden_size)
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

        if language_pretrained_model_path is None:
            print("Warning: no pretrained language model found, training from scratch!!!")

        self.vlbert = VisualLinguisticBert(
            config.NETWORK.VLBERT,
            language_pretrained_model_path=None if config.NETWORK.VLBERT.from_scratch else language_pretrained_model_path
        )

        # init weights
        self.init_weight()

        self.fix_params()

    def init_weight(self):
        if self.config.NETWORK.IMAGE_FEAT_PRECOMPUTED or (not self.config.NETWORK.MASK_RAW_PIXELS):
            self.object_mask_visual_embedding.weight.data.fill_(0.0)
        if self.config.NETWORK.WITH_MVRC_LOSS:
            self.object_mask_word_embedding.weight.data.normal_(mean=0.0, std=self.config.NETWORK.VLBERT.initializer_range)
        self.aux_text_visual_embedding.weight.data.normal_(mean=0.0, std=self.config.NETWORK.VLBERT.initializer_range)
        self.image_feature_extractor.init_weight()
        if self.object_linguistic_embeddings is not None:
            self.object_linguistic_embeddings.weight.data.normal_(mean=0.0,
                                                                  std=self.config.NETWORK.VLBERT.initializer_range)

    def train(self, mode=True):
        super(ResNetVLBERTForAttentionVis, self).train(mode)
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

    def forward(self,
                image,
                boxes,
                im_info,
                text,
                relationship_label,
                mlm_labels,
                mvrc_ops,
                mvrc_labels,
                *aux):

        # concat aux texts from different dataset
        # assert len(aux) > 0 and len(aux) % 2 == 0
        aux_text_list = aux[0::2]
        aux_text_mlm_labels_list = aux[1::2]
        num_aux_text = sum([_text.shape[0] for _text in aux_text_list])
        max_aux_text_len = max([_text.shape[1] for _text in aux_text_list]) if len(aux_text_list) > 0 else 0
        aux_text = text.new_zeros((num_aux_text, max_aux_text_len))
        aux_text_mlm_labels = mlm_labels.new_zeros((num_aux_text, max_aux_text_len)).fill_(-1)
        _cur = 0
        for _text, _mlm_labels in zip(aux_text_list, aux_text_mlm_labels_list):
            _num = _text.shape[0]
            aux_text[_cur:(_cur + _num), :_text.shape[1]] = _text
            aux_text_mlm_labels[_cur:(_cur + _num), :_text.shape[1]] = _mlm_labels
            _cur += _num

        ###########################################

        # visual feature extraction
        images = image
        box_mask = (boxes[:, :, 0] > -1.5)
        origin_len = boxes.shape[1]
        max_len = int(box_mask.sum(1).max().item())
        box_mask = box_mask[:, :max_len]
        boxes = boxes[:, :max_len]
        mvrc_ops = mvrc_ops[:, :max_len]
        mvrc_labels = mvrc_labels[:, :max_len]

        if self.config.NETWORK.IMAGE_FEAT_PRECOMPUTED:
            box_features = boxes[:, :, 4:]
            box_features[mvrc_ops == 1] = self.object_mask_visual_embedding.weight[0]
            boxes[:, :, 4:] = box_features

        obj_reps = self.image_feature_extractor(images=images,
                                                boxes=boxes,
                                                box_mask=box_mask,
                                                im_info=im_info,
                                                classes=None,
                                                segms=None,
                                                mvrc_ops=mvrc_ops,
                                                mask_visual_embed=self.object_mask_visual_embedding.weight[0]
                                                if (not self.config.NETWORK.IMAGE_FEAT_PRECOMPUTED)
                                                   and (not self.config.NETWORK.MASK_RAW_PIXELS)
                                                else None)

        ############################################

        # prepare text
        text_input_ids = text
        text_tags = text.new_zeros(text.shape)
        text_visual_embeddings = self._collect_obj_reps(text_tags, obj_reps['obj_reps'])

        object_linguistic_embeddings = self.object_linguistic_embeddings(
            boxes.new_zeros((boxes.shape[0], boxes.shape[1])).long()
        )
        if self.config.NETWORK.WITH_MVRC_LOSS:
            object_linguistic_embeddings[mvrc_ops == 1] = self.object_mask_word_embedding.weight[0]
        object_vl_embeddings = torch.cat((obj_reps['obj_reps'], object_linguistic_embeddings), -1)

        # add auxiliary text
        max_text_len = max(text_input_ids.shape[1], aux_text.shape[1])
        text_input_ids_multi = text_input_ids.new_zeros((text_input_ids.shape[0] + aux_text.shape[0], max_text_len))
        text_input_ids_multi[:text_input_ids.shape[0], :text_input_ids.shape[1]] = text_input_ids
        text_input_ids_multi[text_input_ids.shape[0]:, :aux_text.shape[1]] = aux_text
        text_token_type_ids_multi = text_input_ids_multi.new_zeros(text_input_ids_multi.shape)
        text_mask_multi = (text_input_ids_multi > 0)
        text_visual_embeddings_multi = text_visual_embeddings.new_zeros((text_input_ids.shape[0] + aux_text.shape[0],
                                                                         max_text_len,
                                                                         text_visual_embeddings.shape[-1]))
        text_visual_embeddings_multi[:text_visual_embeddings.shape[0], :text_visual_embeddings.shape[1]] \
            = text_visual_embeddings
        text_visual_embeddings_multi[text_visual_embeddings.shape[0]:] = self.aux_text_visual_embedding.weight[0]
        object_vl_embeddings_multi = object_vl_embeddings.new_zeros((text_input_ids.shape[0] + aux_text.shape[0],
                                                                     *object_vl_embeddings.shape[1:]))
        object_vl_embeddings_multi[:object_vl_embeddings.shape[0]] = object_vl_embeddings
        box_mask_multi = box_mask.new_zeros((text_input_ids.shape[0] + aux_text.shape[0], *box_mask.shape[1:]))
        box_mask_multi[:box_mask.shape[0]] = box_mask


        ###########################################

        # Visual Linguistic BERT

        encoder_layers, _, attention_probs = self.vlbert(text_input_ids_multi,
                                                         text_token_type_ids_multi,
                                                         text_visual_embeddings_multi,
                                                         text_mask_multi,
                                                         object_vl_embeddings_multi,
                                                         box_mask_multi,
                                                         output_all_encoded_layers=True,
                                                         output_attention_probs=True)
        hidden_states = torch.stack(encoder_layers, dim=0).transpose(0, 1).contiguous()
        attention_probs = torch.stack(attention_probs, dim=0).transpose(0, 1).contiguous()

        return {'attention_probs': attention_probs,
                'hidden_states': hidden_states}




