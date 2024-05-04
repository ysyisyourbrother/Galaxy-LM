
import copy
import math
import os
import warnings
from typing import List, Optional, Tuple, Union

import torch
from torch import nn,Tensor
from galaxy.adapters.utils import modify_model_for_peft
from galaxy.models.bart.bart_model import shift_tokens_right,_expand_mask,_expand_mask,BartEncoderLayer,BartDecoderLayer,BartLearnedPositionalEmbedding,BartEncoder,BartDecoder





class BartPPModel( nn.Module ):
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]

    def __init__(self, config):
        super().__init__( )
        self.config = config
        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        if config.is_encoder:
            encoder_config = copy.deepcopy(config)
            encoder_config.encoder_layers = encoder_config.num_pp_encoder_layers
            if config.is_encoder_first:
                self.encoder = BartEncoder(encoder_config, self.shared)
            else:
                self.encoder = BartEncoder(encoder_config, None)
        if config.is_decoder:
            decoder_config = copy.deepcopy(config)
            decoder_config.decoder_layers = decoder_config.num_pp_decoder_layers
            if config.is_decoder_first:
                self.decoder = BartDecoder(config, self.shared)
            else:
                self.decoder = BartDecoder(config,None)


    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
    ) :
        if self.config.is_encoder:
            # encoder only
            if self.config.is_encoder_first : # 第一层
                assert input_ids is not None
                encoder_outputs = self.encoder(input_ids=input_ids)
            else:
                assert inputs_embeds is not None # 不是第一层
                encoder_outputs = self.encoder(
                    input_ids = None,
                    inputs_embeds=inputs_embeds,
                )
            return encoder_outputs 
        else:  # decoder only
            assert encoder_outputs != None
        if self.config.is_decoder_first : # 第一层
            decoder_input_ids = torch.zeros([self.config.batch_size,1], dtype=torch.long).to(self.config.device) #TODO: 先这样
            decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_outputs ,
            inputs_embeds=None,
            )
            return decoder_outputs
        else:
            assert decoder_inputs_embeds is not None
            decoder_outputs = self.decoder(
                    input_ids = None,
                    inputs_embeds=decoder_inputs_embeds,
                    encoder_hidden_states=encoder_outputs,
                )
            return decoder_outputs
        
class  StageModel(nn.Module):
    def __init__(self, config):
        super(StageModel, self).__init__()
        self.config = config
        # full / lora / adapter
        self.base_model = BartPPModel(config)
        modify_model_for_peft(self.base_model, config)
        if config.is_last_stage: # 最后一个stage，有FC 分类层
            self.lm_head = nn.Linear(config.d_model, config.num_classes, bias=False)
    def forward(self, x):
        if self.config.is_encoder:
            if self.config.is_encoder_first :  # 第一层 x 是input数据
                context = (x[0]).to(self.config.device)
                encoder_outputs = self.base_model(input_ids = context)
                return encoder_outputs
            else:
                 # x只有前面encoder hidden_states
                inputs_embeds = x[0]
                assert inputs_embeds is not None
                encoder_outputs = self.base_model(
                                                    input_ids = None, 
                                                    inputs_embeds = inputs_embeds)
                return encoder_outputs
        else:
            decoder_input_ids = None
            if self.config.is_decoder_first:
                decoder_input_ids = torch.zeros([self.config.batch_size,1], dtype=torch.long).to(self.config.device)
            decoder_inputs_embeds = x[0]
            encoder_outputs = x[1]
            decoder_outputs  = self.base_model(
                input_ids=None,
                decoder_input_ids = decoder_input_ids,
                encoder_outputs = encoder_outputs,
                inputs_embeds = None,
                decoder_inputs_embeds = decoder_inputs_embeds
            )
            if self.config.is_last_stage: # 最后一个stage，有FC 分类层
                pooled = decoder_outputs[:,0,:]
                out = self.lm_head(pooled)
                return out
            else:
                return   decoder_outputs      