
import math
from typing import List, Optional, Tuple, Union
from torch import Tensor, nn
import torch
import torch.utils.checkpoint
from torch import nn
from galaxy.models.llama.llama_model import  _make_causal_mask,_expand_mask
from galaxy.models.llama.llama_model import LlamaRMSNorm,LlamaDecoderLayer
from galaxy.adapters.utils import modify_model_for_peft,get_parameter_number


class PPLlamaModel(nn.Module):
    def __init__(self, config):
        super(PPLlamaModel, self).__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        if self.config.is_first_stage:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([LlamaDecoderLayer(config) for _ in range(config.num_pp_hidden_layers)]) #PP 
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)


    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    # Copied from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape, inputs_embeds.dtype, past_key_values_length=past_key_values_length
            ).to(self.config.device)

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
                self.config.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds = None,
    ) :
        batch_size = self.config.batch_size
        seq_length =  self.config.pad_size
        attention_mask = torch.ones(
            (batch_size, 1, seq_length, seq_length),  device= self.config.device
        )# [bs, 1, seq, seq]
        if self.config.is_first_stage:
            assert self.embed_tokens != None
            inputs_embeds = self.embed_tokens(input_ids)#[bs,seq,hidden_size]
        assert(inputs_embeds is not None)
        hidden_states = inputs_embeds
        for idx, decoder_layer in enumerate(self.layers):
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
            )
            hidden_states = layer_outputs 
        hidden_states = self.norm(hidden_states)
        # pool
        if self.config.is_last_stage:
            pooled_output = hidden_states[:, 0]
            return pooled_output
        else:
            return hidden_states



class  StageModel(nn.Module):
    def __init__(self, config):
        super(StageModel, self).__init__()
        self.config = config
        self.base_model = PPLlamaModel(config)
        if config.is_last_stage: # 最后一个stage，有FC 分类层
            self.lm_head = nn.Linear(config.hidden_size, config.num_classes)
        modify_model_for_peft(self.base_model, config)
    def forward(self, x):
        # x: (token_ids, int(label), seq_len, mask)
        if self.config.is_first_stage: # 第一个stage
            context = (x[0]).to(self.config.device)
            # mask = (x[2]).to(self.config.device)
            hidden_states= self.base_model(
            input_ids=context,
            attention_mask=None,
            inputs_embeds = None
            )
            return hidden_states
        elif self.config.is_last_stage: #最后一个stage 经过分类层
            pooled = self.base_model(
            input_ids=None, 
            attention_mask=None,
            inputs_embeds = x,
            )
            out = self.lm_head(pooled)
            return out
        else: #中间stage
            hidden_states= self.base_model(
            input_ids=None, 
            attention_mask=None,
            inputs_embeds = x,
            )
            return hidden_states