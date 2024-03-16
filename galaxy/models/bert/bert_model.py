import torch
import torch.nn as nn
import math
import copy
from galaxy.loralib.layers import  Linear as LoraLinear



def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        # Bert 原文中采用了三种Embeddings组合方式：Word embeddings+Position embedding+Token type embedding
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
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


class BertMLP(nn.Module):
    def __init__(self, config):
        super(BertMLP, self).__init__()
        self.dense1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.dense2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.activation = gelu
    
    def forward(self, hidden_states):
        return self.dense2(self.dropout(self.activation(self.dense1(hidden_states))))

class BertConnectLayer(nn.Module):
    '''Connective Block: Dropout + Add + Layernorm  '''
    def __init__(self, config):
        super(BertConnectLayer, self).__init__()
        self.ln = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    def forward(self, input, hidden_states):
        hidden_states =  self.dropout(hidden_states)
        hidden_states = self.ln(hidden_states + input)
        return hidden_states
        
        
class BertLayer(nn.Module):
    '''
    Attention --> Connective Block --> MLP --> Connective Block
    '''
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.mlp = BertMLP(config)
        self.con1 = BertConnectLayer(config)
        self.con2 = BertConnectLayer(config)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(  hidden_states , attention_mask)
        connective_output = self.con1(hidden_states ,attention_output)
        mlp_output = self.mlp(connective_output)
        layer_output =  self.con2(connective_output,mlp_output)
        return layer_output


class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        layer = BertLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask):
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
        return hidden_states


class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        # [bs,seq_len,hidden_size] -> [bs,hidden_size]
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

    

class BertModel(nn.Module):
    def __init__(self, config):
        super(BertModel, self).__init__()
        self.config = config

        # 预处理阶段
        self.embeddings = BertEmbeddings(config)
        # 主干网络
        # TODO: Encoder 和 Decoder 可以统一为包含多个TransformerLayer的TransformerBlock
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
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
        hidden_states = self.encoder(embedding_output,
                                      extended_attention_mask)
        # encoder的最终输出结果
        sequence_output = hidden_states
        pooled_output = self.pooler(sequence_output)
        return    pooled_output
