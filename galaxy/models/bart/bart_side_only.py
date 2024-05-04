# coding=utf-8
# Copyright 2021 The Fairseq Authors and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch BART model."""
import copy
import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from galaxy.models.bart.bart_model import BartEncoderLayer,BartDecoderLayer,  _expand_mask,_make_causal_mask,shift_tokens_right


class BartEncoder(nn.Module):
    def __init__(self, config, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__( )
        self.config =  config
        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = config.d_model
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        # self.embed_tokens = nn.Embedding(config.vocab_size, embed_dim, self.padding_idx)

        # if embed_tokens is not None:
        #     self.embed_tokens.weight = embed_tokens.weight

        # self.embed_positions = BartLearnedPositionalEmbedding(
        #     config.max_position_embeddings,
        #     embed_dim,
        # )
        # self.layers = nn.ModuleList([BartEncoderLayer(config) for _ in range(config.encoder_layers)])
        # self.layernorm_embedding = nn.LayerNorm(embed_dim)

        self.gradient_checkpointing = False
        ###########################################################################################
        side_config = copy.deepcopy(config)
        side_config.d_model = side_config.d_model // side_config.side_reduction_factor
        side_config.decoder_ffn_dim = side_config.decoder_ffn_dim // side_config.side_reduction_factor
        side_config.encoder_ffn_dim = side_config.encoder_ffn_dim // side_config.side_reduction_factor
        self.side_layers = nn.ModuleList([BartEncoderLayer(side_config) for _ in range(side_config.encoder_layers)])
        self.side_first_downsample = nn.Linear(config.d_model, side_config.d_model, bias=side_config.add_bias_sampling)
        self.side_downsamples =   nn.ModuleList(
            [  nn.Linear(config.d_model, side_config.d_model, bias=side_config.add_bias_sampling) for i in range(config.encoder_layers )]   
        )
        
    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ) :
        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input = input_ids
            input_ids = input_ids.view(-1, input_ids.shape[-1])
        elif inputs_embeds is not None:
            input = inputs_embeds[:, :, -1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        # if inputs_embeds is None:
        #     inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        # embed_pos = self.embed_positions(input)
        # embed_pos = embed_pos.to(inputs_embeds.device)

        # hidden_states = inputs_embeds + embed_pos
        # hidden_states = self.layernorm_embedding(hidden_states)
        # hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = torch.rand([self.config.batch_size,
                                    self.config.pad_size, 
                                    self.config.d_model], dtype=torch.float32).to(self.config.device)
        ##############################
        side_hidden_states =  self.side_first_downsample(hidden_states)
        ##############################
        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, inputs_embeds.dtype)
        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            if head_mask.size()[0] != (len(self.layers)):
                raise ValueError(
                    f"The head_mask should be specified for {len(self.layers)} layers, but it is for"
                    f" {head_mask.size()[0]}."
                )

        for idx, _ in enumerate(self.side_layers):
            # layer_outputs = encoder_layer(
            #     hidden_states,
            #     attention_mask,
            #     layer_head_mask=(head_mask[idx] if head_mask is not None else None),
            # )
            hidden_states = torch.rand([self.config.batch_size,
                                    self.config.pad_size, 
                                    self.config.d_model], dtype=torch.float32).to(self.config.device)
            side_hidden_states =  side_hidden_states + self.side_downsamples[idx](hidden_states)
            side_layer_outputs =  self.side_layers[idx](
                    side_hidden_states,
                    attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
            )
            side_hidden_states = side_layer_outputs
        return hidden_states,side_hidden_states


class BartDecoder( nn.Module ):

    def __init__(self, config, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__( )
        self.config= config
        
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

        # self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, self.padding_idx)

        # if embed_tokens is not None:
        #     self.embed_tokens.weight = embed_tokens.weight

        # self.embed_positions = BartLearnedPositionalEmbedding(
        #     config.max_position_embeddings,
        #     config.d_model,
        # )
        # self.layers = nn.ModuleList([BartDecoderLayer(config) for _ in range(config.decoder_layers)])
        # self.layernorm_embedding = nn.LayerNorm(config.d_model)

        self.gradient_checkpointing = False
        ####################################################
        side_config = copy.deepcopy(config)
        side_config.d_model = side_config.d_model // side_config.side_reduction_factor
        side_config.decoder_ffn_dim = side_config.decoder_ffn_dim // side_config.side_reduction_factor
        side_config.encoder_ffn_dim = side_config.encoder_ffn_dim // side_config.side_reduction_factor

        self.side_layers = nn.ModuleList([BartDecoderLayer(side_config) for _ in range(config.decoder_layers)])
        self.side_first_downsample = nn.Linear(config.d_model, side_config.d_model, bias=side_config.add_bias_sampling)
        self.side_downsamples =   nn.ModuleList(
            [  nn.Linear(config.d_model, side_config.d_model, bias=side_config.add_bias_sampling) for i in range(config.decoder_layers )]   
        )

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        side_encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    )  :
        # retrieve input_ids and inputs_embeds
        
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input = input_ids
            input_shape = input.shape
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            input = inputs_embeds[:, :, -1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        # if inputs_embeds is None:
            # inputs_embeds = self.embed_tokens(input) * self.embed_scale
            


        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, input_shape, inputs_embeds, past_key_values_length
        )

        # expand encoder attention mask
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            encoder_attention_mask = _expand_mask(encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])

        # embed positions
        # positions = self.embed_positions(input, past_key_values_length)
        # positions = positions.to(inputs_embeds.device)

        # hidden_states = inputs_embeds + positions
        # hidden_states = self.layernorm_embedding(hidden_states)

        # hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = torch.rand([self.config.batch_size,
                                    1, 
                                    self.config.d_model], dtype=torch.float32).to(self.config.device)
        ################################################################################################
        side_hidden_states = self.side_first_downsample(hidden_states)
        #################################################################################################
        # check if head_mask/cross_attn_head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip([head_mask, cross_attn_head_mask], ["head_mask", "cross_attn_head_mask"]):
            if attn_mask is not None:
                if attn_mask.size()[0] != (len(self.layers)):
                    raise ValueError(
                        f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for"
                        f" {head_mask.size()[0]}."
                    )

        for idx, _ in enumerate(self.side_layers):
            past_key_value = past_key_values[idx] if past_key_values is not None else None
            # layer_outputs = decoder_layer(
            #         hidden_states,
            #         attention_mask=attention_mask,
            #         encoder_hidden_states=encoder_hidden_states,
            #         encoder_attention_mask=encoder_attention_mask,
            #         layer_head_mask=(head_mask[idx] if head_mask is not None else None),
            #         cross_attn_layer_head_mask=(
            #             cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None
            #         ),
            #         past_key_value=past_key_value,
            #     )
            hidden_states = torch.rand([self.config.batch_size,
                                    1, 
                                    self.config.d_model], dtype=torch.float32).to(self.config.device)
            side_hidden_states =  side_hidden_states + self.side_downsamples[idx](hidden_states)
            side_layer_outputs = self.side_layers[idx](
                        side_hidden_states,
                        attention_mask=attention_mask,
                        encoder_hidden_states=side_encoder_hidden_states,
                         layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    cross_attn_layer_head_mask=(
                        cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None
                    ),
                    past_key_value=past_key_value,)
            side_hidden_states = side_layer_outputs
        return hidden_states, side_hidden_states



class BartModel( nn.Module ):
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]

    def __init__(self, config):
        super().__init__( )
        self.config = config
        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        # self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.encoder = BartEncoder(config,None)
        self.decoder = BartDecoder(config, None)
        self.side_final_upsample = nn.Linear(config.d_model // config.side_reduction_factor, config.d_model, bias=config.add_bias_sampling)

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
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
    )  :
        # different to other models, Bart automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided
        decoder_input_ids = torch.zeros([self.config.batch_size,1], dtype=torch.long).to(self.config.device) #TODO: 先这样
        hidden_states, side_hidden_states = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
            )
        
        decoder_outputs,side_decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=hidden_states,
            side_encoder_hidden_states=side_hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
        )
        out = decoder_outputs + self.side_final_upsample(side_decoder_outputs)
        return out
