import torch
import torch.nn as nn
import math
import copy
from galaxy.models.bert.bert_model import gelu, swish
from galaxy.models.bert.bert_model import BertLayerNorm, BertEmbeddings,BertPooler, BertMLP
from galaxy.models.bert.tp_bert_model import TPBertAttention,TPBertMLP
from galaxy.models.bert.sp_bert_model import SPBertAttention
from galaxy.core.model_parallel.mappings import (
    copy_to_tensor_model_parallel_region,
    scatter_to_sequence_parallel_region,
    gather_from_sequence_parallel_region,
)

class GalaxyBertConnectLayer(nn.Module):
    def __init__(self, config,is_first):
        super(GalaxyBertConnectLayer, self).__init__()
        self.ln = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.config = config
        self.is_first = is_first
    def forward(self, input, hidden_states):
        hidden_states =  self.dropout(hidden_states)
        hidden_states = self.ln(hidden_states + input)
        # CON 没有TP
        if not hasattr(self.config, 'con_parallel_method') or self.config.con_parallel_method == "None": # ATT(TP) -- CON -- MLP(TP) -- CON
            # print("Nothing in CON1")
            return hidden_states
        elif self.config.con_parallel_method == "TP":
            raise NotImplementedError("CON can not be TP")
        # ATT -- CON1 (SP) -- MLP -- CON2 (SP)
        if self.is_first: # Next block is MLP
            if self.config.mlp_parallel_method == "TP": #  SP -- AllGather -- TP
                return gather_from_sequence_parallel_region(hidden_states, True ,self.config.seq_scatter_list)
            elif self.config.mlp_parallel_method == "SP": # SP -- Nothing -- SP
                return hidden_states
        else: # 下一个block 是ATT
            if self.config.att_parallel_method == "TP": # SP -- AllGather -- TP
                return gather_from_sequence_parallel_region(hidden_states, True,self.config.seq_scatter_list)
            elif self.config.att_parallel_method == "SP": # SP -- Nothing -- SP
                return hidden_states # SP -- Nothing -- SP TODO: 这里需要确认一下
  

class GalaxyBertLayer(nn.Module):
    ''''
        ATT-- CON 1 (SP) -- MLP -- CON 2 (SP)
    '''
    def __init__(self, config):
        super(GalaxyBertLayer, self).__init__()
        if config.att_parallel_method == "TP":
            self.attention = TPBertAttention(config) # 根据后续CON是SP还是NONE,使用 ReduceScatter 还是 AllReduce
        elif config.att_parallel_method == "SP":
            self.attention = SPBertAttention(config)
        if config.mlp_parallel_method == "TP":
            self.mlp = TPBertMLP(config) #  根据后续CON是SP还是NONE,使用 ReduceScatter 还是 AllReduce
        elif config.mlp_parallel_method == "SP":
            self.mlp = BertMLP(config) # SP的MLP后面接一定是SP的CON，所以不需要通信
        # 区分第一个和第二个   ATT-- CON 1 -- MLP -- CON 2  
        self.con1 = GalaxyBertConnectLayer(config,is_first=True) 
        self.con2 = GalaxyBertConnectLayer(config,is_first=False)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(  hidden_states , attention_mask)
        inputs = torch.zeros(attention_output.size(),dtype=attention_output.dtype,device=attention_output.device) # TODO: hidden_states 在CON的时候SP， forward的时候 也是要划分的,先不管
        connective_output = self.con1(inputs ,attention_output)
        mlp_output = self.mlp(connective_output)
        inputs = torch.zeros(mlp_output.size(),dtype=mlp_output.dtype,device=mlp_output.device)
        layer_output =  self.con2(inputs, mlp_output)
        return layer_output

class GalaxyBertEncoder(nn.Module):
    """ 很多个TPBertLayer堆叠成为TPBertEncoder """
    def __init__(self, config):
        super(GalaxyBertEncoder, self).__init__()
        layer = GalaxyBertLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask ):
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
        return hidden_states
    
    
class GalaxyBertModel(nn.Module):
    def __init__(self, config):
        super(GalaxyBertModel, self).__init__()
        self.config = config

        # 预处理阶段
        self.embeddings = BertEmbeddings(config)
        # 主干网络
        # TODO: Encoder 和 Decoder 可以统一为包含多个TransformerLayer的TransformerBlock
        self.encoder = GalaxyBertEncoder(config)
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
        # ATT 不同的并行方式 决定最初的输入 TP: copy to all ; SP: split along dim  ;
        if self.config.att_parallel_method == "TP":
            tp_input = copy_to_tensor_model_parallel_region(embedding_output) 
            hidden_states = self.encoder(tp_input,
                                      extended_attention_mask,
                                    )
        else: 
           #TODO:
            scatter_sp_input = scatter_to_sequence_parallel_region(embedding_output, self.config.seq_scatter_list)
            hidden_states = self.encoder(scatter_sp_input,
                                      extended_attention_mask,
                                   )
        
        # encoder的最终输出结果
        sequence_output = hidden_states
        #  最终的输出是否需要gather
        if  self.config.att_parallel_method == "SP" :
            sequence_output =  gather_from_sequence_parallel_region(sequence_output, False,
                                                                self.config.seq_scatter_list)
        pooled_output = self.pooler(sequence_output)
       
        return pooled_output