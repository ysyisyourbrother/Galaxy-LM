import torch
import torch.nn as nn
import math
import copy
from galaxy.models.t5.t5_model import  T5Block,T5LayerNorm,_expand_mask
from typing import List, Optional, Tuple, Union
from galaxy.adapters.utils import  mark_only_side_as_trainable
from torch import Tensor
class T5SidePPStack( nn.Module):
    def __init__(self, config, embed_tokens=None):
        super().__init__( )
        self.config=config
        self.dtype = torch.float32
        self.embed_tokens = embed_tokens
        self.is_decoder = config.is_decoder
        #######################################
        self.block = nn.ModuleList(
            [T5Block(config, has_relative_attention_bias=bool(i == 0)) for i in range(config.num_layers)]
        )
        if config.is_encoder_last or config.is_decoder_last:
            self.final_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)
        #######################################
        side_config = copy.deepcopy(config)
        side_config.d_ff = side_config.d_ff // self.config.side_reduction_factor
        side_config.d_kv = side_config.d_kv // self.config.side_reduction_factor
        side_config.d_model = side_config.d_model // self.config.side_reduction_factor
        self.side_block = nn.ModuleList(
            [T5Block(side_config, has_relative_attention_bias=bool(i == 0)) for i in range(config.num_layers)]
        )
        if config.is_encoder_first or config.is_decoder_first:
            self.side_first_downsample = nn.Linear(config.d_model, side_config.d_model,  bias=config.add_bias_sampling)
        self.side_downsamples = nn.ModuleList( [ nn.Linear(config.d_model, side_config.d_model,  bias=config.add_bias_sampling) for  _ in range(config.num_layers)] )
        if config.is_encoder_last or config.is_decoder_last:
            self.final_side_layer_norm = T5LayerNorm(side_config.d_model, eps=side_config.layer_norm_epsilon)
    def _convert_head_mask_to_5d(self, head_mask, num_hidden_layers):
        """-> [num_hidden_layers x batch x num_heads x seq_length x seq_length]"""
        if head_mask.dim() == 1:
            head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
        elif head_mask.dim() == 2:
            head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
        assert head_mask.dim() == 5, f"head_mask.dim != 5, instead {head_mask.dim()}"
        head_mask = head_mask.to(dtype=self.dtype)  # switch to float if need + fp16 compatibility
        return head_mask
    def invert_attention_mask(self, encoder_attention_mask: Tensor) -> Tensor:
        if encoder_attention_mask.dim() == 3:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
        if encoder_attention_mask.dim() == 2:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
        encoder_extended_attention_mask = encoder_extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * torch.finfo(self.dtype).min

        return encoder_extended_attention_mask
    def get_head_mask(
        self, head_mask: Optional[Tensor], num_hidden_layers: int, is_attention_chunked: bool = False
    ) -> Tensor:
        if head_mask is not None:
            head_mask = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
            if is_attention_chunked is True:
                head_mask = head_mask.unsqueeze(-1)
        else:
            head_mask = [None] * num_hidden_layers

        return head_mask
    def forward(
        self,
        input_ids=None,
        inputs_embeds=None,
        side_inputs_embeds=None,
        encoder_hidden_states=None,
        side_encoder_hidden_states=None,
    ):
        encoder_attention_mask = None
        head_mask = None
        cross_attn_head_mask = None
        if input_ids is not None and inputs_embeds is not None:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You cannot specify both {err_msg_prefix}input_ids and {err_msg_prefix}inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(f"You have to specify either {err_msg_prefix}input_ids or {err_msg_prefix}inputs_embeds")

        if inputs_embeds is None:
            if self.embed_tokens is None:
                raise ValueError("You have to initialize the model with valid token embeddings")
        # 经过embdding
            inputs_embeds = self.embed_tokens(input_ids)
        batch_size, seq_length = input_shape
        # required mask seq length can be calculated via length of past
        mask_seq_length =  seq_length
        past_key_values = [None] * len(self.block)
        attention_mask = torch.ones(batch_size, mask_seq_length, device=inputs_embeds.device)
        # extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)
        extended_attention_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
                self.config.device
            )
        
        if self.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(
                    encoder_hidden_shape, device=inputs_embeds.device, dtype=torch.long
                )
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None
        # Prepare head mask if needed

        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        cross_attn_head_mask = self.get_head_mask(cross_attn_head_mask, self.config.num_layers)

        # side block first input
        if self.config.is_encoder_first or self.config.is_decoder_first:
            hidden_states = self.dropout(inputs_embeds)
            side_hidden_states = self.side_first_downsample(hidden_states)
        else:
            hidden_states = inputs_embeds
            side_hidden_states = side_inputs_embeds
        # 经过N layer
        for i, (layer_module, _) in enumerate(zip(self.block, past_key_values)):
            layer_head_mask = head_mask[i]
            cross_attn_layer_head_mask = cross_attn_head_mask[i]
            layer_outputs = layer_module(
                hidden_states,
                attention_mask=extended_attention_mask,
                position_bias=None,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
                encoder_decoder_position_bias=None,
                layer_head_mask=layer_head_mask,
                cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=None,
                use_cache=False,
                output_attentions=False,
            )
            hidden_states  = layer_outputs[0]
            side_hidden_states = side_hidden_states + self.side_downsamples[i](hidden_states)
            side_layer_outputs = self.side_block[i](
                side_hidden_states,
                position_bias=None,
                encoder_hidden_states = side_encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
                encoder_decoder_position_bias=None,
                layer_head_mask=layer_head_mask,
                cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=None,
                use_cache=False,
                output_attentions=False,
            )
            side_hidden_states = side_layer_outputs[0]
        if self.config.is_encoder_last or  self.config.is_decoder_last :
            hidden_states = self.final_layer_norm(hidden_states)
            hidden_states = self.dropout(hidden_states)
            
            side_hidden_states = self.final_side_layer_norm(side_hidden_states)
            side_hidden_states = self.dropout(side_hidden_states)
        return hidden_states, side_hidden_states
    
class T5SidePPModel(nn.Module):
    _keys_to_ignore_on_load_unexpected = [
        "decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight",
    ]
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]

    def __init__(self, config):
        super().__init__( )
        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        self.config = config
        #################################################################

        if config.is_encoder:
            encoder_config = copy.deepcopy(config)
            encoder_config.is_decoder = False
            encoder_config.use_cache = False
            encoder_config.is_encoder_decoder = False
            if config.is_encoder_first:
                self.encoder = T5SidePPStack(encoder_config, self.shared)
            else:
                self.encoder = T5SidePPStack(encoder_config, None)
        if config.is_decoder:
            decoder_config = copy.deepcopy(config)
            decoder_config.is_decoder = True
            decoder_config.is_encoder_decoder = False
            decoder_config.num_layers = config.num_decoder_layers
            if config.is_decoder_first:
                self.decoder = T5SidePPStack(decoder_config, self.shared)
            else:
                self.decoder = T5SidePPStack(decoder_config, None)
        #############################################
            if config.is_decoder_last:
                self.side_final_upsample =  nn.Linear(config.d_model //  config.side_reduction_factor, 
                                                    config.d_model, 
                                                    bias=config.add_bias_sampling)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        side_inputs_embeds: Optional[torch.Tensor] = None,
        decoder_inputs_embeds: Optional[torch.Tensor] = None,
        side_decoder_inputs_embeds: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        side_encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
    ):
        if self.config.is_encoder: #encoder only 
            if self.config.is_encoder_first :
                assert input_ids is not None
                encoder_outputs, side_encoder_outputs =self.encoder(
                        input_ids=input_ids,)
            else:
                assert inputs_embeds is not None # 不是第一层
                assert side_inputs_embeds is not None
                encoder_outputs, side_encoder_outputs = self.encoder(
                input_ids=None,
                inputs_embeds=inputs_embeds,
                side_inputs_embeds=side_inputs_embeds,
            )
            return encoder_outputs, side_encoder_outputs
        
        else: # decoder only 
            assert encoder_outputs is not None
            assert side_encoder_outputs is not None
            if self.config.is_decoder_first : # 第一层
                assert decoder_input_ids is not None
                decoder_outputs, side_decoder_outputs = self.decoder(
                    input_ids = decoder_input_ids,
                    inputs_embeds = None,
                    side_inputs_embeds = None,
                    encoder_hidden_states = encoder_outputs,
                    side_encoder_hidden_states = side_encoder_outputs
                    )
            else:
                assert decoder_inputs_embeds is not None
                assert side_decoder_inputs_embeds is not None
                decoder_outputs,side_decoder_outputs = self.decoder(
                    input_ids = None,
                    inputs_embeds = decoder_inputs_embeds,
                    side_inputs_embeds = side_decoder_inputs_embeds,
                    encoder_hidden_states = encoder_outputs,
                    side_encoder_hidden_states = side_encoder_outputs
                    )
            if self.config.is_decoder_last : 
                sequence_output =  self.side_final_upsample(side_encoder_outputs)
                sequence_output =  sequence_output + decoder_outputs
                return sequence_output,None
            else:
                return decoder_outputs, side_decoder_outputs


class SideStageModel(nn.Module):
    def __init__(self, config):
        super(SideStageModel, self).__init__()
        self.config = config
        self.base_model = T5SidePPModel(config)
        mark_only_side_as_trainable(self.base_model)
        if config.is_last_stage: # 最后一个stage，有FC 分类层
            self.lm_head = nn.Linear(config.d_model, config.num_classes, bias=False)
            
    def forward(self,x):
        if self.config.is_encoder:
            if self.config.is_encoder_first :  # 第一层 x 是input数据
                context = (x[0]).to(self.config.device)
                encoder_outputs,side_encoder_outputs = self.base_model(input_ids = context)
                return encoder_outputs,side_encoder_outputs
            else:
                # 前面层的hidden_state, side_hidden_state [hidden_state,side_hidden_state]
                inputs_embeds = x[0]
                side_inputs_embeds = x[1]
                assert inputs_embeds is not None
                assert side_inputs_embeds is not None
                encoder_outputs,side_encoder_outputs = self.base_model(input_ids = None, 
                                                                        inputs_embeds = inputs_embeds,
                                                                        side_inputs_embeds = side_inputs_embeds)
                return encoder_outputs,side_encoder_outputs
        else: # decoder only x = [decoder_inputs_embeds, side_decoder_inputs_embeds, encoder_outputs, side_encoder_outputs]
            if self.config.is_decoder_first:
                decoder_input_ids = torch.zeros([self.config.batch_size,1], dtype=torch.long).to(self.config.device)
                decoder_outputs,side_decoder_outputs = self.base_model(
                    input_ids=None,
                    decoder_input_ids = decoder_input_ids,
                    encoder_outputs = x[2],
                    side_encoder_outputs = x[3],
                )
            else:
                assert x[0] is not None
                assert x[1] is not None
                decoder_outputs,side_decoder_outputs = self.base_model(
                    input_ids = None,
                    decoder_input_ids = None,
                    decoder_inputs_embeds = x[0],
                    side_decoder_inputs_embeds = x[1],
                    encoder_outputs = x[2],
                    side_encoder_outputs = x[3]
                )
            if self.config.is_decoder_last: # 最后一个stage，有FC 分类层
                pooled = decoder_outputs[:,0,:]
                out = self.lm_head(pooled)
                return out
            else:
                return   decoder_outputs ,side_decoder_outputs

            