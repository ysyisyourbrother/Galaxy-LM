
import torch
import torch.nn as nn
import copy
from galaxy.models.bert.bert_model import gelu, swish
from galaxy.models.bert.bert_model import BertEmbeddings,BertPooler,BertMLP,BertConnectLayer,BertLayer
from galaxy.loralib.layers import  Linear as LoraLinear

class PPBertEncoder(nn.Module):
    def __init__(self, config):
        super(PPBertEncoder, self).__init__()
        layer = BertLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_pp_hidden_layers)])

    def forward(self, hidden_states, attention_mask ):

        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
        return hidden_states
class PPBertModel(nn.Module):
    def __init__(self, config):
        super(PPBertModel, self).__init__()
        self.config = config

        # 预处理阶段
        if config.is_first_stage:
            self.embeddings = BertEmbeddings(config)
        # 主干网络
        # TODO: Encoder 和 Decoder 可以统一为包含多个TransformerLayer的TransformerBlock
        self.encoder = PPBertEncoder(config)

        if config.is_last_stage:
            self.pooler = BertPooler(config)

    def forward(self, input_ids, encoder_input=None, token_type_ids=None, attention_mask=None ):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        # 第一个stage,经过embedding 
        if self.config.is_first_stage:
            encoder_input = self.embeddings(input_ids, token_type_ids)

        assert encoder_input is not None
        hidden_states = self.encoder(encoder_input,
                                    extended_attention_mask,)
        if self.config.is_last_stage:
            sequence_output = hidden_states
            pooled_output = self.pooler(sequence_output)
            return   pooled_output
        else:
            return hidden_states
