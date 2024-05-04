import torch
import torch.nn as nn
import copy
from galaxy.models.bert.bert_model import  BertEmbeddings,BertPooler,BertLayer
from galaxy.models.bert.bert_side_model import mark_only_side_as_trainable 

class SidePPBertEncoder(nn.Module):
    def __init__(self, config):
        super(SidePPBertEncoder, self).__init__()
        layer = BertLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_pp_hidden_layers)])
        # update side_config 
        side_config = copy.deepcopy(config)
        side_config.hidden_size = int (side_config.hidden_size // side_config.side_reduction_factor)
        side_config.intermediate_size = int  (side_config.intermediate_size // side_config.side_reduction_factor)
        side_config.att_head_size = int (side_config.hidden_size // side_config.num_attention_heads)
        side_layer = BertLayer(side_config)
        self.side_layer =  nn.ModuleList([copy.deepcopy(side_layer) for _ in range(config.num_pp_hidden_layers)])
        side_downsamples = nn.Linear(config.hidden_size , side_config.hidden_size)
        self.side_downsamples =  nn.ModuleList( [copy.deepcopy(side_downsamples) for _ in 
                                                range(config.num_pp_hidden_layers)])
    def forward(self, hidden_states,
                side_hidden_states,
                attention_mask, 
                ):
        for i, layer_module in enumerate(self.layer):
            hidden_states = layer_module(hidden_states, attention_mask)
            side_hidden_states = side_hidden_states + self.side_downsamples[i](hidden_states)
            side_hidden_states = self.side_layer[i](side_hidden_states, attention_mask)
        return hidden_states, side_hidden_states

class SidePPBertModel(nn.Module):
    def __init__(self, config):
        super(SidePPBertModel, self).__init__()
        self.config = config

        # 预处理阶段
        if  config.is_first_stage:
            self.embeddings = BertEmbeddings(config)
            self.side_first_downsample = nn.Linear(config.hidden_size, config.hidden_size // config.side_reduction_factor)
        # 主干网络
        # TODO: Encoder 和 Decoder 可以统一为包含多个TransformerLayer的TransformerBlock
        self.encoder = SidePPBertEncoder(config)
        
        if config.is_last_stage:
            self.side_final_upsample = nn.Linear(config.hidden_size // config.side_reduction_factor, config.hidden_size)
            self.pooler = BertPooler(config)
    def forward(self, input_ids, hidden_state, hidden_state_side, token_type_ids=None, attention_mask=None ):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        # 第一个stage,经过embedding 和 first downsample
        if self.config.is_first_stage:
            hidden_state = self.embeddings(input_ids, token_type_ids)
            hidden_state_side = self.side_first_downsample(hidden_state)
        assert hidden_state is not None
        assert hidden_state_side is not None
        
        hidden_states,side_hidden_states = self.encoder(hidden_state,
                                     hidden_state_side,
                                      extended_attention_mask,
                                       )
        if self.config.is_last_stage: # 最后一个stage 得到最终结果
            sequence_output = hidden_states + self.side_final_upsample(side_hidden_states)
            pooled_output = self.pooler(sequence_output)
            return    pooled_output 
        else:  # 否则返回中间states
            return hidden_states,side_hidden_states
        
        
class StageModel(nn.Module):
    def __init__(self, config):
        super(StageModel, self).__init__()
        self.base_model = SidePPBertModel(config)
        self.config = config
        if config.is_last_stage: # 最后一个stage，有FC 分类层
            self.fc = nn.Linear(config.hidden_size, config.num_classes)
        mark_only_side_as_trainable(self.base_model)
    def forward(self, x,  side):   
        # x: (token_ids, int(label), seq_len, mask)
        if self.config.is_first_stage: # 第一个stage
            context, mask = x[0].to(self.config.device), x[2].to(self.config.device)
            hidden_states, side_hidden_states = self.base_model(input_ids = context,
                                                        hidden_state = None,
                                                        hidden_state_side = None,
                                                        token_type_ids = None,
                                                        attention_mask = mask
                                                        )
            return hidden_states, side_hidden_states
        elif self.config.is_last_stage: #最后一个stage 经过分类层
            input_ids = torch.zeros(self.config.batch_size, self.config.pad_size).long().to(self.config.device)
            pooled =  self.base_model(input_ids = input_ids,
                                hidden_state = x,
                                hidden_state_side = side,
                                token_type_ids=None, 
                                attention_mask=None)
            out = self.fc(pooled)
            return out
        else:
            input_ids = torch.zeros(self.config.batch_size, self.config.pad_size).long().to(self.config.device)
            hidden_states, side_hidden_states  = self.base_model(input_ids = input_ids,
                                hidden_state = x,
                                hidden_state_side = side,
                                token_type_ids=None, 
                                attention_mask=None)
            return hidden_states, side_hidden_states 
        