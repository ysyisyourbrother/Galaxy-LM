import torch
import torch.nn as nn
import math
import copy
from galaxy.models.bert.bert_model import  BertEmbeddings,BertPooler,BertLayer




def mark_only_side_as_trainable(model: nn.Module)  -> None:
    for n, p in model.named_parameters():
        if 'side_' not in n:
            p.requires_grad = False


class SideBertEncoder(nn.Module):
    def __init__(self, config):
        super(SideBertEncoder, self).__init__()
        layer = BertLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])
        # update side_config 
        side_config = copy.deepcopy(config)
        side_config.hidden_size = int (side_config.hidden_size // side_config.side_reduction_factor)
        side_config.intermediate_size = int  (side_config.intermediate_size // side_config.side_reduction_factor)
        side_config.att_head_size = int (side_config.hidden_size // side_config.num_attention_heads)
        side_layer = BertLayer(side_config)
        self.side_layer =  nn.ModuleList([copy.deepcopy(side_layer) for _ in range(config.num_hidden_layers)])
        side_downsamples = nn.Linear(config.hidden_size , side_config.hidden_size)
        self.side_downsamples =  nn.ModuleList( [copy.deepcopy(side_downsamples) for _ in 
                                                range(config.num_hidden_layers)])
    def forward(self, hidden_states,
                side_hidden_states,
                attention_mask, 
                ):
        for i, layer_module in enumerate(self.layer):
            hidden_states = layer_module(hidden_states, attention_mask)
            side_hidden_states = side_hidden_states + self.side_downsamples[i](hidden_states)
            side_hidden_states = self.side_layer[i](side_hidden_states, attention_mask)
        return hidden_states, side_hidden_states
class SideBertModel(nn.Module):
    def __init__(self, config):
        super(SideBertModel, self).__init__()
        self.config = config

        # 预处理阶段
        self.embeddings = BertEmbeddings(config)
        # 主干网络
        # TODO: Encoder 和 Decoder 可以统一为包含多个TransformerLayer的TransformerBlock
        self.encoder = SideBertEncoder(config)
        
        self.side_first_downsample = nn.Linear(config.hidden_size, config.hidden_size // config.side_reduction_factor)
        self.side_final_upsample = nn.Linear(config.hidden_size // config.side_reduction_factor, config.hidden_size)
        self.pooler = BertPooler(config)
    def forward(self, input_ids, token_type_ids=None, attention_mask=None ):
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

        embedding_output = self.embeddings(input_ids, token_type_ids)
        side_embedding_output = self.side_first_downsample(embedding_output)
        hidden_states,side_hidden_states = self.encoder(embedding_output,
                                     side_embedding_output,
                                      extended_attention_mask,
                                       )
        # encoder的最终输出结果
        sequence_output = hidden_states + self.side_final_upsample(side_hidden_states)
        pooled_output = self.pooler(sequence_output)
        return    pooled_output

