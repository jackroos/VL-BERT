import torch
import torch.nn as nn
from external.pytorch_pretrained_bert.modeling import BertEncoder, BertLayerNorm


class BertEncoderWrapper(nn.Module):
    def __init__(self, bert_config, input_size, output_all_encoded_layers=False):
        super(BertEncoderWrapper, self).__init__()
        self.bert_config = bert_config
        self.output_all_encoded_layers = output_all_encoded_layers
        self.input_transform = nn.Linear(input_size, bert_config.hidden_size)
        self.with_position_embeddings = False if 'with_position_embeddings' not in bert_config \
            else bert_config.with_position_embeddings
        if self.with_position_embeddings:
            self.position_embedding = nn.Embedding(bert_config.max_position_embeddings, bert_config.hidden_size)
            self.LayerNorm = BertLayerNorm(bert_config.hidden_size, eps=1e-12)
            self.dropout = nn.Dropout(bert_config.hidden_dropout_prob)
        self.bert_encoder = BertEncoder(bert_config)

        self.apply(self.init_bert_weights)

    def init_bert_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.bert_config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def get_output_dim(self):
        return self.bert_config.hidden_size

    def forward(self, inputs, mask):
        inputs = self.input_transform(inputs)
        if self.with_position_embeddings:
            seq_length = inputs.size(1)
            position_ids = torch.arange(seq_length, dtype=torch.long, device=inputs.device)
            position_ids = position_ids.unsqueeze(0).expand((inputs.shape[0], inputs.shape[1]))
            position_embeddings = self.position_embedding(position_ids)
            inputs = inputs + position_embeddings
            inputs = self.LayerNorm(inputs)
            inputs = self.dropout(inputs)

        extended_attention_mask = mask.unsqueeze(1).unsqueeze(2)
        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        output = self.bert_encoder(inputs,
                                   extended_attention_mask,
                                   output_all_encoded_layers=self.output_all_encoded_layers)
        if not self.output_all_encoded_layers:
            output = output[0]
        return output

