import torch
import torch.nn as nn
import math
import copy
from galaxy.models.bert.bert_model import gelu, swish
from galaxy.models.bert.bert_model import BertEmbeddings,BertPooler,BertMLP,BertConnectLayer
from galaxy.loralib.layers import  Linear as LoraLinear
from galaxy.core.model_parallel.mappings import (
    scatter_to_sequence_parallel_region,
    gather_from_sequence_parallel_region,
    reduce_scatter_for_tp_to_sp
)

class SPBertAttention(nn.Module):
    def __init__(self, config):
        super(SPBertAttention, self).__init__()
        # hidden_size = num_attention_heads * head_size
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.config = config
        # SP下此设备实际运行的head数量
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.att_head_size
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 定义qkv大小，考虑张量并行对head的分割。默认qkv head_size相同
        self.qkv_projection_size = self.config.att_head_size * self.num_attention_heads
        if config.use_lora == False or config.lora_dim == 0:
            self.query = nn.Linear(config.hidden_size, self.qkv_projection_size)
            self.key = nn.Linear(config.hidden_size, self.qkv_projection_size)
            self.value = nn.Linear(config.hidden_size, self.qkv_projection_size)
        else:
            self.query = LoraLinear(config.hidden_size, 
                               self.qkv_projection_size,
                               r = config.lora_dim,
                               lora_alpha = config.lora_alpha,
                               lora_dropout = config.lora_dropout,
                               fan_in_fan_out = config.fan_in_fan_out, # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
                               merge_weights = config.merge_weights) 
            self.key = nn.Linear(config.hidden_size, self.qkv_projection_size)
            self.value = LoraLinear(config.hidden_size, 
                               self.qkv_projection_size,
                               r = config.lora_dim,
                               lora_alpha = config.lora_alpha,
                               lora_dropout = config.lora_dropout,
                               fan_in_fan_out = config.fan_in_fan_out, # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
                               merge_weights = config.merge_weights) 
            

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        # Linear output
        self.dense = nn.Linear(self.qkv_projection_size, self.config.hidden_size)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        """
        Arguments:
            hidden_states: [bs, seq_len//sp_group, hidden_size]
            attention_mask: [1,1,1,0,0,...,0] 标记sequence中哪些为有效位置
        """
        # 计算QKV矩阵
        # [bs, seq_len//sp_group, hidden_size] -> [bs, seq_len//sp_group, hidden_size]
        mixed_query_layer = self.query(hidden_states)
        # [bs, seq_len//sp_group, hidden_size] -> [bs, seq_len//sp_group, hidden_size]
        mixed_key_layer = self.key(hidden_states)
        # [bs, seq_len//sp_group, hidden_size] -> [bs, seq_len//sp_group, hidden_size]
        mixed_value_layer = self.value(hidden_states)

        # 插入通信原语：K和V矩阵 all-gather
        # [bs, seq_len//sp_group, hidden_size] -> [bs, seq_len, hidden_size]
        gather_key_layer = gather_from_sequence_parallel_region(mixed_key_layer, False,
                                                                self.config.seq_scatter_list)
        # [bs, seq_len//sp_group, hidden_size] -> [bs, seq_len, hidden_size]
        gather_value_layer = gather_from_sequence_parallel_region(mixed_value_layer, False,
                                                                  self.config.seq_scatter_list)

        # 调整QKV矩阵的shape，使其满足Multi-head Attention形式
        # [bs, seq_len//sp_group, hidden_size] -> [bs, num_att_head, seq_len//sp_group, att_head_size]
        query_layer = self.transpose_for_scores(mixed_query_layer)
        # [bs, seq_len, hidden_size] -> [bs, num_att_head, seq_len, att_head_size]
        key_layer = self.transpose_for_scores(gather_key_layer)
        # [bs, seq_len, hidden_size] -> [bs, num_att_head, seq_len, att_head_size]
        value_layer = self.transpose_for_scores(gather_value_layer)

        # 计算Q*K
        # [bs, num_att_head, seq_len//sp_group, att_head_size] -> [bs, num_att_head, seq_len//sp_group, seq_len]
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
        # [bs, num_att_head, seq_len//sp_group, seq_len] -> [bs, num_att_head, seq_len//sp_group, att_head_size]
        context_layer = torch.matmul(attention_probs, value_layer)
        # [bs, num_att_head, seq_len//sp_group, att_head_size] -> [bs, seq_len//sp_group, num_att_head, att_head_size]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        # [bs, seq_len//sp_group, num_att_head, att_head_size] -> [bs, seq_len//sp_group, hidden_size]
        context_layer = context_layer.view(*new_context_layer_shape)

        # Linear output
        multi_attention_output = self.dense(context_layer)
        # 因为ATT后面接的CON,CON 是 SP,所以这里不需要gather
        return multi_attention_output


class SPBertLayer(nn.Module):
    def __init__(self, config):
        super(SPBertLayer, self).__init__()
        self.attention = SPBertAttention(config)
        self.mlp = BertMLP(config)
        self.con1 = BertConnectLayer(config)
        self.con2 = BertConnectLayer(config)


    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(  hidden_states , attention_mask)
        connective_output = self.con1(hidden_states ,attention_output)
        mlp_output = self.mlp(connective_output)
        layer_output = self.con2(connective_output,mlp_output)
        return layer_output



class SPBertEncoder(nn.Module):
    """ 很多个SPBertLayer堆叠成为SPBertEncoder """
    def __init__(self, config):
        super(SPBertEncoder, self).__init__()
        layer = SPBertLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask ):
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
        return hidden_states




class SPBertModel(nn.Module):
    def __init__(self, config):
        super(SPBertModel, self).__init__()
        self.config = config

        # 预处理阶段
        self.embeddings = BertEmbeddings(config)
        # 主干网络
        # TODO: Encoder 和 Decoder 可以统一为包含多个TransformerLayer的TransformerBlock
        self.encoder = SPBertEncoder(config)
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

        # Scatter input to devices:
        scatter_sp_input = scatter_to_sequence_parallel_region(embedding_output, self.config.seq_scatter_list)
        
        # 序列并行目前只支持输出最后层的结果
        output_all_encoded_layers = False

        hidden_states = self.encoder(scatter_sp_input,
                                      extended_attention_mask,
                                     )
        # encoder的最终输出结果
        sequence_output = hidden_states

        # AllGather output to devices:
        gather_sp_output = gather_from_sequence_parallel_region(sequence_output, False, 
                                                                self.config.seq_scatter_list)
        
        pooled_output = self.pooler(gather_sp_output)
        return  pooled_output
