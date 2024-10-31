# coding=utf-8
# Copyright 2018 Mesh TensorFlow authors, T5 Authors and HuggingFace Inc. team.
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
""" PyTorch T5 model."""


import copy
import math
import os
import warnings
from typing import List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.utils.checkpoint import checkpoint

from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
    Seq2SeqQuestionAnsweringModelOutput,
    Seq2SeqSequenceClassifierOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS, find_pruneable_heads_and_indices, prune_linear_layer
from transformers.utils import (
    DUMMY_INPUTS,
    DUMMY_MASK,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_torch_fx_proxy,
    logging,
    replace_return_docstrings,
)
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map
from transformers.models.t5.configuration_t5 import T5Config
from .modeling_t5 import __HEAD_MASK_WARNING_MSG
from .modeling_t5 import (
    T5Block,
    T5LayerNorm,
    T5Stack,
    T5Model,
    T5ClassificationHead,
    T5PreTrainedModel,
    T5ForSequenceClassification)
logger = logging.get_logger(__name__)
from .QSTConfig import AdapterLinear
from .modeling_qst_output import SideBaseModelOutput,SideBaseModelOutputWithPastAndCrossAttentions,SideSeq2SeqModelOutput

class QSTT5Stack(T5PreTrainedModel):
    def __init__(self, config, llm: T5Stack, embed_tokens, qsconfig):
        super().__init__(config)

        self.embed_tokens = embed_tokens
        self.embed_tokens.weight = llm.embed_tokens.weight
        self.embed_tokens.weight.requires_grad = False
        self.is_decoder = llm.is_decoder
        #######################################
        #  stack
        self.block = llm.block  
        self.final_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.final_layer_norm.weight = llm.final_layer_norm.weight
        self.final_layer_norm.weight.requires_grad = False
        self.dropout = nn.Dropout(config.dropout_rate)
        self.gradient_checkpointing = False
        #############################################################
        # side network
        side_config = copy.deepcopy(config)
        side_config.d_ff = int(side_config.d_ff //qsconfig.r)
        side_config.d_kv = int(side_config.d_kv // qsconfig.r)
        side_config.d_model = int(side_config.d_model //  qsconfig.r)
        self.downsample = nn.ModuleList([AdapterLinear(in_features=config.d_model,
                                                       out_features=int(config.d_model / qsconfig.r),
                                                       r=int(qsconfig.peft_hidden_size),
                                                       alpha_r=int(qsconfig.peft_hidden_size),
                                                       activation=qsconfig.activation,
                                                       # num_expert=QSTConfig.num_expert,
                                                       # routing_strategy=QSTConfig.routing_strategy,
                                                       # weight_average=QSTConfig.weight_average,
                                                       add_layer_norm_after_adapter=qsconfig.add_layer_norm_after_adapter,
                                                       add_layer_norm_before_adapter=qsconfig.add_layer_norm_before_adapter,
                                                       dropout=qsconfig.dropout) #TODO:没用to(self.blackbone[i].mlp.gate_proj.weight.device)
                                         for i in range(side_config.num_layers)])
        self.z = nn.ParameterList([nn.Parameter(torch.tensor([0.5]))  for i in range(side_config.num_hidden_layers)])#TODO:没用to(self.blackbone[i].mlp.gate_proj.weight.device)
        self.side_block = nn.ModuleList([T5Block(side_config, has_relative_attention_bias=bool(i == 0)) for i in range(side_config.num_layers)])
        self.side_final_layer_norm = T5LayerNorm(side_config.d_model, eps=side_config.layer_norm_epsilon)
        #############################################################
        # Initialize weights and apply final processing
        # self.post_init()
        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False

    def parallelize(self, device_map=None):
        warnings.warn(
            "`T5Stack.parallelize` is deprecated and will be removed in v5 of Transformers, you should load your model"
            " with `device_map='balanced'` in the call to `from_pretrained`. You can also provide your own"
            " `device_map` but it needs to be a dictionary module_name to device, so for instance {'block.0': 0,"
            " 'block.1': 1, ...}",
            FutureWarning,
        )
        # Check validity of device_map
        self.device_map = (
            get_device_map(len(self.block), range(torch.cuda.device_count())) if device_map is None else device_map
        )
        assert_device_map(self.device_map, len(self.block))
        self.model_parallel = True
        self.first_device = "cpu" if "cpu" in self.device_map.keys() else "cuda:" + str(min(self.device_map.keys()))
        self.last_device = "cuda:" + str(max(self.device_map.keys()))
        # Load onto devices
        for k, v in self.device_map.items():
            for layer in v:
                cuda_device = "cuda:" + str(k)
                self.block[layer] = self.block[layer].to(cuda_device)

        # Set embed_tokens to first layer
        self.embed_tokens = self.embed_tokens.to(self.first_device)
        # Set final layer norm to last device
        self.final_layer_norm = self.final_layer_norm.to(self.last_device)

    def deparallelize(self):
        warnings.warn(
            "Like `parallelize`, `deparallelize` is deprecated and will be removed in v5 of Transformers.",
            FutureWarning,
        )
        self.model_parallel = False
        self.device_map = None
        self.first_device = "cpu"
        self.last_device = "cpu"
        for i in range(len(self.block)):
            self.block[i] = self.block[i].to("cpu")
        self.embed_tokens = self.embed_tokens.to("cpu")
        self.final_layer_norm = self.final_layer_norm.to("cpu")
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        side_encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        side_past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        # Model parallel
        if self.model_parallel:
            torch.cuda.set_device(self.first_device)
            self.embed_tokens = self.embed_tokens.to(self.first_device)
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

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
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape

        # required mask seq length can be calculated via length of past
        mask_seq_length = past_key_values[0][0].shape[2] + seq_length if past_key_values is not None else seq_length

        if use_cache is True:
            if not self.is_decoder:
                raise ValueError(f"`use_cache` can only be set to `True` if {self} is used as a decoder")

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, mask_seq_length, device=inputs_embeds.device)
        if self.is_decoder and encoder_attention_mask is None and encoder_hidden_states is not None:
            encoder_seq_length = encoder_hidden_states.shape[1]
            encoder_attention_mask = torch.ones(
                batch_size, encoder_seq_length, device=inputs_embeds.device, dtype=torch.long
            )

        # initialize past_key_values with `None` if past does not exist
        if past_key_values is None:
            past_key_values = [None] * len(self.block)
        if side_past_key_values is None:
            side_past_key_values = [None] * len(self.side_block)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=inputs_embeds.device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        cross_attn_head_mask = self.get_head_mask(cross_attn_head_mask, self.config.num_layers)
        present_key_value_states = () if use_cache else None
        present_side_key_value_states = () if use_cache else None

        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None
        
        all_side_hidden_states = () if output_hidden_states else None
        all_side_attentions = () if output_attentions else None
        all_side_cross_attentions = () if (output_attentions and self.is_decoder) else None

        
        position_bias = None
        encoder_decoder_position_bias = None

        side_position_bias = None
        side_encoder_decoder_position_bias = None

        # backbone_hidden_state = None
        side_past_key_value = None
        
        hidden_states = self.dropout(inputs_embeds)
        # first downsample 
        side_hidden_states = self.downsample[0](hidden_states)
        side_present_key_value_state = None
        for i, (layer_module, past_key_value) in enumerate(zip(self.block, past_key_values)):
            
            layer_head_mask = head_mask[i]
            cross_attn_layer_head_mask = cross_attn_head_mask[i]
            
            side_layer_module = self.side_block[i]
            side_past_key_value = side_past_key_values[i]

            # Model parallel
            # if self.model_parallel:
            #     #TODO: Not implemented yet
            #     torch.cuda.set_device(hidden_states.device)
            #     # Ensure that attention_mask is always on the same device as hidden_states
            #     if attention_mask is not None:
            #         attention_mask = attention_mask.to(hidden_states.device)
            #     if position_bias is not None:
            #         position_bias = position_bias.to(hidden_states.device)
            #     if encoder_hidden_states is not None:
            #         encoder_hidden_states = encoder_hidden_states.to(hidden_states.device)
            #     if encoder_extended_attention_mask is not None:
            #         encoder_extended_attention_mask = encoder_extended_attention_mask.to(hidden_states.device)
            #     if encoder_decoder_position_bias is not None:
            #         encoder_decoder_position_bias = encoder_decoder_position_bias.to(hidden_states.device)
            #     if layer_head_mask is not None:
            #         layer_head_mask = layer_head_mask.to(hidden_states.device)
            #     if cross_attn_layer_head_mask is not None:
            #         cross_attn_layer_head_mask = cross_attn_layer_head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
                all_side_hidden_states = all_side_hidden_states + (side_hidden_states,)

            if self.gradient_checkpointing and self.training:
                pass
                #TODO: Not implemented yet
                # def create_custom_forward(module):
                #     def custom_forward(*inputs):
                #         return tuple(module(*inputs, use_cache, output_attentions))

                #     return custom_forward

                # layer_outputs = checkpoint(
                #     create_custom_forward(layer_module),
                #     hidden_states,
                #     extended_attention_mask,
                #     position_bias,
                #     encoder_hidden_states,
                #     encoder_extended_attention_mask,
                #     encoder_decoder_position_bias,
                #     layer_head_mask,
                #     cross_attn_layer_head_mask,
                #     None,  # past_key_value is always None with gradient checkpointing
                # )
            else:
                with torch.no_grad():
                    layer_outputs = layer_module(
                        hidden_states,
                        attention_mask=extended_attention_mask,
                        position_bias=position_bias,
                        encoder_hidden_states=encoder_hidden_states,
                        encoder_attention_mask=encoder_extended_attention_mask,
                        encoder_decoder_position_bias=encoder_decoder_position_bias,
                        layer_head_mask=layer_head_mask,
                        cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                        past_key_value=past_key_value,
                        use_cache=use_cache,
                        output_attentions=output_attentions,
                    )
                hidden_states = layer_outputs[0]
                z = torch.sigmoid(self.z[i])
                side_hidden_states = (1 - z) * self.downsample[i](hidden_states) + z * side_hidden_states
                side_layer_outputs = side_layer_module(
                    side_hidden_states,
                    attention_mask=extended_attention_mask,
                    position_bias=side_position_bias,
                    encoder_hidden_states=side_encoder_hidden_states,
                    encoder_attention_mask=encoder_extended_attention_mask,
                    encoder_decoder_position_bias=side_encoder_decoder_position_bias,
                    layer_head_mask=layer_head_mask,
                    cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                    past_key_value=side_past_key_value,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    
                )
          
            # layer_outputs is a tuple with:
            # hidden-states, key-value-states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)
            if use_cache is False:
                layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]
                side_layer_outputs = side_layer_outputs[:1] + (None,) + side_layer_outputs[1:]
            hidden_states, present_key_value_state = layer_outputs[:2]
            side_hidden_states, side_present_key_value_state = side_layer_outputs[:2]
            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, key-value-states (self-attention position bias), (self-attention weights),
            # (cross-attention position bias), (cross-attention weights)
            position_bias = layer_outputs[2]
            side_position_bias = side_layer_outputs[2]
            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]
                side_encoder_decoder_position_bias = side_layer_outputs[4 if output_attentions else 3]
            # append next layer key value states
            if use_cache: 
                present_key_value_states = present_key_value_states + (present_key_value_state,)
                present_side_key_value_states = present_side_key_value_states + (side_present_key_value_state,)
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[3],)
                all_side_attentions = all_side_attentions + (side_layer_outputs[3],)
                if self.is_decoder:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[5],)
                    all_side_cross_attentions = all_side_cross_attentions + (side_layer_outputs[5],)


            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                # TODO: Not implemented yet
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        side_hidden_states = self.side_final_layer_norm(side_hidden_states)
        side_hidden_states = self.dropout(side_hidden_states)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
            all_side_hidden_states = all_side_hidden_states + (side_hidden_states,)


        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states, 
                    present_key_value_states,
                    all_hidden_states,
                    all_attentions,
                    all_cross_attentions,
                    
                    side_hidden_states,
                    present_side_key_value_states,
                    all_side_hidden_states,
                    all_side_attentions,
                    all_side_cross_attentions,
                ]
                if v is not None
            )
        return SideBaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=present_key_value_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
            
            side_last_hidden_state=side_hidden_states,
            side_past_key_values=present_side_key_value_states,
            side_hidden_states=all_side_hidden_states,
            side_attentions=all_side_attentions,
            side_cross_attentions=all_side_cross_attentions,
        )
    def save_qst_state(self, path):
        qst_layers_path = os.path.join(path, "qst_layers_parameters.pt")
        torch.save(self.side_block.state_dict(), qst_layers_path)

        qst_z_path = os.path.join(path, "qst_z_parameters.pt")
        torch.save(self.z.state_dict(), qst_z_path)
        
        
        qst_downsample_path = os.path.join(path, "qst_downsample_parameters.pt")
        torch.save(self.downsample.state_dict(), qst_downsample_path)
        
        qst_norm_path = os.path.join(path, "qst_norm_parameters.pt")
        torch.save(self.side_final_layer_norm.state_dict(), qst_norm_path)
        
class QSTT5Model(T5PreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [
        "decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight",
    ]
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]

    def __init__(self,  llm:T5Model,  config: T5Config, qstconfig):
        super().__init__(config)
        # embedding
        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        self.shared.weight = llm.shared.weight
        self.shared.weight.requires_grad = False
        
        # encoder layers
        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = QSTT5Stack(encoder_config, llm.encoder, self.shared, qstconfig)
        
        # decoder layers
        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = QSTT5Stack(decoder_config, llm.decoder, self.shared, qstconfig)

        # Initialize weights and apply final processing
        # self.post_init()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    def parallelize(self, device_map=None):
        warnings.warn(
            "`T5Model.parallelize` is deprecated and will be removed in v5 of Transformers, you should load your model"
            " with `device_map='balanced'` in the call to `from_pretrained`. You can also provide your own"
            " `device_map` but it needs to be a dictionary module_name to device, so for instance {'encoder.block.0':"
            " 0, 'encoder.block.1': 1, ...}",
            FutureWarning,
        )
        self.device_map = (
            get_device_map(len(self.encoder.block), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.encoder.block))
        self.encoder.parallelize(self.device_map)
        self.decoder.parallelize(self.device_map)
        self.model_parallel = True

    def deparallelize(self):
        
        warnings.warn(
            "Like `parallelize`, `deparallelize` is deprecated and will be removed in v5 of Transformers.",
            FutureWarning,
        )
        self.encoder.deparallelize()
        self.decoder.deparallelize()
        self.encoder = self.encoder.to("cpu")
        self.decoder = self.decoder.to("cpu")
        self.model_parallel = False
        self.device_map = None
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        side_past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        decoder_inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqModelOutput]:
        r"""
        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, T5Model

        >>> tokenizer = AutoTokenizer.from_pretrained("t5-small")
        >>> model = T5Model.from_pretrained("t5-small")

        >>> input_ids = tokenizer(
        ...     "Studies have been shown that owning a dog is good for you", return_tensors="pt"
        ... ).input_ids  # Batch size 1
        >>> decoder_input_ids = tokenizer("Studies show that", return_tensors="pt").input_ids  # Batch size 1

        >>> # preprocess: Prepend decoder_input_ids with start token which is pad token for T5Model.
        >>> # This is not needed for torch's T5ForConditionalGeneration as it does this internally using labels arg.
        >>> decoder_input_ids = model._shift_right(decoder_input_ids)

        >>> # forward pass
        >>> outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
        >>> last_hidden_states = outputs.last_hidden_state
        ```"""
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, SideBaseModelOutput):
            
            encoder_outputs = SideBaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
        
                side_last_hidden_state=encoder_outputs.side_last_hidden_state,
                side_hidden_states=encoder_outputs.side_hidden_states,
                side_attentions=encoder_outputs.side_attentions
            )
        hidden_states = encoder_outputs[0]
        side_hidden_states = encoder_outputs.side_last_hidden_state
        

        # Set device for model parallelism
        # if self.model_parallel:
        #     torch.cuda.set_device(self.decoder.first_device)
        #     hidden_states = hidden_states.to(self.decoder.first_device)
        #     if decoder_input_ids is not None:
        #         decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
        #     if attention_mask is not None:
        #         attention_mask = attention_mask.to(self.decoder.first_device)
        #     if decoder_attention_mask is not None:
        #         decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            side_encoder_hidden_states=side_hidden_states,
            side_past_key_values=side_past_key_values,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return SideSeq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
            
            side_last_hidden_state=decoder_outputs.side_last_hidden_state,
            side_past_key_values=decoder_outputs.side_past_key_values,
            side_decoder_hidden_states=decoder_outputs.side_hidden_states,
            side_decoder_attentions=decoder_outputs.side_attentions,
            side_cross_attentions=decoder_outputs.side_cross_attentions,
            side_encoder_last_hidden_state=encoder_outputs.side_last_hidden_state,
            side_encoder_hidden_states=encoder_outputs.side_hidden_states,
            side_encoder_attentions=encoder_outputs.side_attentions,
        )

    def save_qst_state(self, path):
        self.encoder.save_qst_state(path)
        self.decoder.save_qst_state(path)


class QSTT5ForSequenceClassification(T5PreTrainedModel):
    _keys_to_ignore_on_load_unexpected = ["decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight"]
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]

    def __init__(self,llm:T5ForSequenceClassification, config: T5Config,qstconfig):
        super().__init__(config)
        self.merge_final_lst=qstconfig.merge_final_lst
        self.hidden_size = config.d_model
        self.transformer = QSTT5Model(llm.transformer, config,qstconfig)
        self.classification_head = T5ClassificationHead(config)
        self.qstconfig = qstconfig
        if qstconfig.merge_final_lst:
            # self.side_gate_params_merge_last = nn.Parameter(torch.ones(1) * qstconfig.gate_alpha) 
            self.lm_head_z = nn.Parameter(torch.ones(self.hidden_size) )
        else:
            self.lm_head_z = nn.Parameter(torch.zeros(self.hidden_size))
        self.upsample = nn.Linear(int(self.hidden_size / qstconfig.r), self.hidden_size) 
        del llm
        # Initialize weights and apply final processing
        self.post_init()

        self.model_parallel = False

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
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Seq2SeqSequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        Returns:
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if labels is not None:
            use_cache = False

        if input_ids is None and inputs_embeds is not None:
            raise NotImplementedError(
                f"Passing input embeddings is currently not supported for {self.__class__.__name__}"
            )

        # Copied from models.bart.modeling_bart.BartModel.forward different to other models, T5 automatically creates
        # decoder_input_ids from input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            if input_ids is None:
                raise ValueError(
                    "If no `decoder_input_ids` or `decoder_inputs_embeds` are "
                    "passed, `input_ids` cannot be `None`. Please pass either "
                    "`input_ids` or `decoder_input_ids` or `decoder_inputs_embeds`."
                )
            decoder_input_ids = self._shift_right(input_ids)

        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs.last_hidden_state
        side_hidden_states  = outputs.side_last_hidden_state
        side_hidden_states = self.upsample(side_hidden_states)
        
        if self.merge_final_lst:
            gate = torch.sigmoid(self.lm_head_z )
            final_hidden_states = gate* side_hidden_states + (1 - gate) * hidden_states
        else:
            lm_head_z = torch.sigmoid(self.lm_head_z)
            final_hidden_states = lm_head_z * side_hidden_states + (1 - lm_head_z) * hidden_states
        eos_mask = input_ids.eq(self.config.eos_token_id).to(hidden_states.device)

        if len(torch.unique_consecutive(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")
        batch_size, _, hidden_size = final_hidden_states.shape
        sentence_representation = final_hidden_states[eos_mask, :].view(batch_size, -1, hidden_size)[:, -1, :]
        logits = self.classification_head(sentence_representation)

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.config.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.config.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.config.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    def save_qst_state(self, path):
        pass 
        # self.transformer.save_qst_state(path)
        # qst_upsample_path = os.path.join(path, "qst_upsample_parameters.pt")
        # torch.save(self.upsample.state_dict(), qst_upsample_path)
        
        # lm_head_z_path = os.path.join(path, "lm_head_z_parameters.pt")
        # torch.save(self.lm_head_z, lm_head_z_path)