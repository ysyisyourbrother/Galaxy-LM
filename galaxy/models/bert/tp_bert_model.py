import torch
import torch.nn as nn
import math
import copy
from galaxy.models.bert.bert_model import gelu, swish
from galaxy.models.bert.bert_model import BertLayerNorm, BertEmbeddings,BertPooler,BertConnectLayer
from galaxy.core.model_parallel.mappings import (
    copy_to_tensor_model_parallel_region,
    reduce_from_tensor_model_parallel_region,
    reduce_scatter_for_tp_to_sp
)

class TPBertAttention(nn.Module):
    def __init__(self, config):
        super(TPBertAttention, self).__init__()
        # hidden_size = num_attention_heads * head_size
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.config = config
        # TP下此设备实际运行的head数量
        self.num_attention_heads = config.tp_num_attention_heads
        self.attention_head_size = config.att_head_size
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 定义qkv大小，考虑张量并行对head的分割。默认qkv head_size相同
        self.qkv_projection_size = self.config.att_head_size * self.config.tp_num_attention_heads

        self.query = nn.Linear(config.hidden_size, self.qkv_projection_size)
        self.key = nn.Linear(config.hidden_size, self.qkv_projection_size)
        self.value = nn.Linear(config.hidden_size, self.qkv_projection_size)

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
            hidden_states: [bs, seq_len, hidden_size]
            attention_mask: [1,1,1,0,0,...,0] 标记sequence中哪些为有效位置
        """
        # TODO: 实现到core.tp中去，可以通过配置决定线性操作是否需要在开头allreduce或者在结尾allgather

        # 计算QKV矩阵
        # [bs, seq_len, hidden_size] -> [bs, seq_len, hidden_size//tp_gourp]
        mixed_query_layer = self.query(hidden_states)
        # [bs, seq_len, hidden_size] -> [bs, seq_len, hidden_size//tp_gourp]
        mixed_key_layer = self.key(hidden_states)
        # [bs, seq_len, hidden_size] -> [bs, seq_len, hidden_size//tp_gourp]
        mixed_value_layer = self.value(hidden_states)

        # 调整QKV矩阵的shape，使其满足Multi-head Attention形式
        # [bs, seq_len, hidden_size] -> [bs, num_att_head//tp_group, seq_len, att_head_size]
        query_layer = self.transpose_for_scores(mixed_query_layer)
        # [bs, seq_len, hidden_size] -> [bs, num_att_head//tp_group, seq_len, att_head_size]
        key_layer = self.transpose_for_scores(mixed_key_layer)
        # [bs, seq_len, hidden_size] -> [bs, num_att_head//tp_group, seq_len, att_head_size]
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # 计算Q*K
        # [bs, num_att_head, seq_len, att_head_size] -> [bs, num_att_head//tp_group, seq_len, seq_len]
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
        # [bs, num_att_head, seq_len, seq_len] -> [bs, num_att_head//tp_group, seq_len, att_head_size]
        context_layer = torch.matmul(attention_probs, value_layer)
        # [bs, num_att_head, seq_len, att_head_size] -> [bs, seq_len, num_att_head//tp_group, att_head_size]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        # [bs, seq_len, num_att_head, att_head_size] -> [bs, seq_len, hidden_size//tp_group]
        context_layer = context_layer.view(*new_context_layer_shape)

        # Linear output
        multi_attention_output = self.dense(context_layer)
        # 根据下一个CON是怎么并行的，插入张量并行通信原语
        if not hasattr(self.config, 'con_parallel_method') or  self.config.con_parallel_method == "None": # CON不并行，all reduce
            return reduce_from_tensor_model_parallel_region(multi_attention_output)
        elif self.config.con_parallel_method == "SP": # CON SP, reduce scatter
            return  reduce_scatter_for_tp_to_sp(multi_attention_output, self.config.seq_scatter_list)
        else:
            raise NotImplementedError("con_parallel_method should be SP or None")
            
 

class TPBertMLP(nn.Module):
    def __init__(self, config):
        super(TPBertMLP, self).__init__()
        self.config = config
        self.dense1 = nn.Linear(config.hidden_size, config.tp_intermediate_size)
        self.dense2 = nn.Linear(config.tp_intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.activation = gelu
    
    def forward(self, hidden_states):
        mlp_output = self.dense2(self.dropout(self.activation(self.dense1(hidden_states))))
        # 根据下一个CON是怎么并行的，插入张量并行通信原语
        if not hasattr(self.config, 'con_parallel_method') or  self.config.con_parallel_method == "None": # CON不并行，all reduce
            return reduce_from_tensor_model_parallel_region(mlp_output)
        elif self.config.con_parallel_method == "SP": #CON SP, reduce scatter
            return reduce_scatter_for_tp_to_sp(mlp_output, self.config.seq_scatter_list)
        else:
            raise NotImplementedError("con_parallel_method should be SP or None")



class TPBertLayer(nn.Module):
    """
    ATT(TP) -- CON 1 -- MLP(TP) -- CON 2
    """
    def __init__(self, config):
        super(TPBertLayer, self).__init__()
        self.attention = TPBertAttention(config)
        self.mlp = TPBertMLP(config)
        self.con1 = BertConnectLayer(config)
        self.con2 = BertConnectLayer(config)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(  hidden_states , attention_mask)
        connective_output = self.con1(hidden_states ,attention_output)
        mlp_output = self.mlp(connective_output)
        layer_output =  self.con2(connective_output,mlp_output)
        return layer_output



class TPBertEncoder(nn.Module):
    """ 很多个TPBertLayer堆叠成为TPBertEncoder """
    def __init__(self, config):
        super(TPBertEncoder, self).__init__()
        layer = TPBertLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        """
        Arguments:
            output_all_encoded_layers: 是否要保存每一层BertLayer的输出结果
        """
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class TPBertModel(nn.Module):
    def __init__(self, config):
        super(TPBertModel, self).__init__()
        self.config = config

        # 预处理阶段
        self.embeddings = BertEmbeddings(config)
        # 主干网络
        # TODO: Encoder 和 Decoder 可以统一为包含多个TransformerLayer的TransformerBlock
        self.encoder = TPBertEncoder(config)
        self.pooler = BertPooler(config)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, output_all_encoded_layers=True):
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
        # Copy input to devices
        tp_input = copy_to_tensor_model_parallel_region(embedding_output)
        encoded_layers = self.encoder(tp_input,
                                      extended_attention_mask,
                                      output_all_encoded_layers=output_all_encoded_layers)
        # encoder的最终输出结果
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers, pooled_output
