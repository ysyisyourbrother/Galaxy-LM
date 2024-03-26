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
from transformers.activations import ACT2FN
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



logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "T5Config"
_CHECKPOINT_FOR_DOC = "google-t5/t5-small"

####################################################
# This dict contains ids and associated url
# for the pretrained weights provided with the models
####################################################
T5_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "google-t5/t5-small",
    "google-t5/t5-base",
    "google-t5/t5-large",
    "google-t5/t5-3b",
    "google-t5/t5-11b",
    # See all T5 models at https://huggingface.co/models?filter=t5
]
from galaxy.models.t5.t5_model import T5PreTrainedModel
from galaxy.models.t5.t5_model import T5Stack

class T5PPModel(T5PreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [
        "decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight",
    ]
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        if config.is_encoder:
            encoder_config = copy.deepcopy(config)
            encoder_config.is_decoder = False
            encoder_config.use_cache = False
            encoder_config.num_layers = config.num_pp_encoder_layers
            encoder_config.is_encoder_decoder = False
            if config.is_encoder_first:
                self.encoder = T5Stack(encoder_config, self.shared)
            else:
                self.encoder = T5Stack(encoder_config, None)
        if config.is_decoder:
            decoder_config = copy.deepcopy(config)
            decoder_config.is_decoder = True
            decoder_config.use_cache = False
            decoder_config.is_encoder_decoder = False
            decoder_config.num_layers = config.num_pp_decoder_layers
            if config.is_decoder_first:
                self.decoder = T5Stack(decoder_config, self.shared)
            else:
                self.decoder = T5Stack(decoder_config, None)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        decoder_inputs_embeds: Optional[torch.Tensor] = None,
    ):
        if self.config.is_encoder:
            # encoder only
            if self.config.is_encoder_first : # 第一层
                assert input_ids is not None
                encoder_outputs = self.encoder(input_ids=input_ids)
            else:
                assert inputs_embeds is not None # 不是第一层
                # print(input_ids)
                # print(inputs_embeds.shape)
                encoder_outputs = self.encoder(
                    inputs_embeds=inputs_embeds,
                )
            return encoder_outputs 
        else: # decoder only
            assert encoder_outputs != None
            if self.config.is_decoder_first : # 第一层
                assert decoder_input_ids is not None
                decoder_outputs = self.decoder(
                    input_ids=decoder_input_ids,
                    inputs_embeds=None,
                    encoder_hidden_states=encoder_outputs,
                )
                return  decoder_outputs
            else:
                assert decoder_inputs_embeds is not None
                decoder_outputs = self.decoder(
                    input_ids = None,
                    inputs_embeds=decoder_inputs_embeds,
                    encoder_hidden_states=encoder_outputs,
                )
                return decoder_outputs
        
        
