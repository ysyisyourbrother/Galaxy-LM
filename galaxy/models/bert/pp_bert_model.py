from json import encoder
import torch
import torch.nn as nn
import math
import copy
from galaxy.models.bert.bert_model import gelu, swish
from galaxy.models.bert.bert_model import BertEmbeddings,BertPooler,BertMLP,BertConnectLayer
from galaxy.loralib.layers import  Linear as LoraLinear
 
class PPBertAttention(nn.Module):
    def __init__(self, config):
        super(PPBertAttention, self).__init__()
        # hidden_size = num_attention_heads * head_size
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        if config.use_lora == False or config.lora_att_dim == 0:
            self.query = nn.Linear(config.hidden_size, self.all_head_size)
            self.key = nn.Linear(config.hidden_size, self.all_head_size)
            self.value = nn.Linear(config.hidden_size, self.all_head_size)
        else:
            self.query = LoraLinear(config.hidden_size, 
                               self.all_head_size,
                               r = config.lora_att_dim,
                               lora_alpha = config.lora_alpha,
                               lora_dropout = config.lora_dropout,
                               fan_in_fan_out = config.fan_in_fan_out, # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
                               merge_weights = config.merge_weights) 
            self.key = nn.Linear(config.hidden_size, self.all_head_size)
            self.value = LoraLinear(config.hidden_size, 
                                self.all_head_size,
                                r = config.lora_att_dim,
                                lora_alpha = config.lora_alpha,
                                lora_dropout = config.lora_dropout,
                                fan_in_fan_out = config.fan_in_fan_out, # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
                                merge_weights = config.merge_weights) 
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)    
        # Linear output
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        """
        Arguments:
            hidden_states: [bs, seq_len, hidden_size]
            attention_mask: [1,1,1,0,0,...,0] 标记sequence中哪些为有效位置
        """
        # 计算QKV矩阵
        # [bs, seq_len, hidden_size] -> [bs, seq_len, hidden_size]
        mixed_query_layer = self.query(hidden_states)
        # [bs, seq_len, hidden_size] -> [bs, seq_len, hidden_size]
        mixed_key_layer = self.key(hidden_states)
        # [bs, seq_len, hidden_size] -> [bs, seq_len, hidden_size]
        mixed_value_layer = self.value(hidden_states)

        # 调整QKV矩阵的shape，使其满足Multi-head Attention形式
        # [bs, seq_len, hidden_size] -> [bs, num_att_head, seq_len, att_head_size]
        query_layer = self.transpose_for_scores(mixed_query_layer)
        # [bs, seq_len, hidden_size] -> [bs, num_att_head, seq_len, att_head_size]
        key_layer = self.transpose_for_scores(mixed_key_layer)
        # [bs, seq_len, hidden_size] -> [bs, num_att_head, seq_len, att_head_size]
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # 计算Q*K
        # [bs, num_att_head, seq_len, att_head_size] -> [bs, num_att_head, seq_len, seq_len]
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask
        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # 计算(QK)*V
        # [bs, num_att_head, seq_len, seq_len] -> [bs, num_att_head, seq_len, att_head_size]
        context_layer = torch.matmul(attention_probs, value_layer)
        # [bs, num_att_head, seq_len, att_head_size] -> [bs, seq_len, num_att_head, att_head_size]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        # [bs, seq_len, num_att_head, att_head_size] -> [bs, seq_len, hidden_size]
        context_layer = context_layer.view(*new_context_layer_shape)
        
        # Linear output
        multi_attention_output = self.dense(context_layer)
        return multi_attention_output


 


class PPBertLayer(nn.Module):
    def __init__(self, config):
        super(PPBertLayer, self).__init__()
        self.attention = PPBertAttention(config)
        self.mlp = BertMLP(config)
        self.con1 = BertConnectLayer(config)
        self.con2 = BertConnectLayer(config)


    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(  hidden_states , attention_mask)
        connective_output = self.con1(hidden_states ,attention_output)
        mlp_output = self.mlp(connective_output)
        layer_output = self.con2(connective_output,mlp_output)
        return layer_output


class PPBertEncoder(nn.Module):
    def __init__(self, config):
        super(PPBertEncoder, self).__init__()
        layer = PPBertLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=False):
        """
        Arguments:
            output_all_encoded_layers: 是否要保存每一个BertLayer的输出结果
        """
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers

 


class PPBertModel(nn.Module):
    def __init__(self, config):
        super(PPBertModel, self).__init__()
        self.config = config

        # 预处理阶段
        if config.pre_process:
            self.embeddings = BertEmbeddings(config)
        # 主干网络
        # TODO: Encoder 和 Decoder 可以统一为包含多个TransformerLayer的TransformerBlock
        self.encoder = PPBertEncoder(config)

        if config.post_process:
            self.pooler = BertPooler(config)

    def forward(self, input_ids, encoder_input=None, token_type_ids=None, attention_mask=None, output_all_encoded_layers=True):
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

        if self.config.pre_process:
            encoder_input = self.embeddings(input_ids, token_type_ids)

        assert encoder_input is not None
        encoded_layers = self.encoder(encoder_input,
                                      extended_attention_mask,
                                      output_all_encoded_layers=output_all_encoded_layers)

        if self.config.post_process:
            sequence_output = encoded_layers[-1]
            pooled_output = self.pooler(sequence_output)
            if not output_all_encoded_layers:
                encoded_layers = encoded_layers[-1]
            return encoded_layers, pooled_output
        else:
            if not output_all_encoded_layers:
                encoded_layers = encoded_layers[-1]
            return encoded_layers, None
