import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import argparse

from  train_config.llama.pp_llama_config import PPLlamaConfig
from galaxy.models.llama.pp_llama_model import PPLlamaModel
from galaxy.data.build import build_dataset, build_iterator,get_time_dif
from galaxy.initialize import initialize_galaxy,get_args
from galaxy.core.pipeline_parallel.schedules import PipelineRuntime
from galaxy.loralib.utils import mark_only_lora_as_trainable, get_parameter_number
from galaxy.tokenizer.tokenizer import BertTokenizer
from galaxy.utils import clean_up, get_max_memory

class  StageModel(nn.Module):
    def __init__(self, config):
        super(StageModel, self).__init__()
        self.config = config
        self.llama_model = PPLlamaModel(config)
        if config.is_last_stage: # 最后一个stage，有FC 分类层
            self.lm_head = nn.Linear(config.hidden_size, config.num_classes)
        if not config.use_lora or config.lora_att_dim == 0:
            print("not use lora, train full parameters ... ")
            for param in self.llama_model.parameters():
                param.requires_grad = True
        else:
            print("use lora, mark_only_lora_as_trainable ... ")
            mark_only_lora_as_trainable(self.llama_model)
    def forward(self, x):
        # x: (token_ids, int(label), seq_len, mask)
        if self.config.is_first_stage: # 第一个stage
            context = (x[0]).to(self.config.device)
            # mask = (x[2]).to(self.config.device)
            hidden_states= self.llama_model(
            input_ids=context,
            attention_mask=None,
            inputs_embeds = None
            )
            return hidden_states
        elif self.config.is_last_stage: #最后一个stage 经过分类层
            pooled = self.llama_model(
            input_ids=None, 
            attention_mask=None,
            inputs_embeds = x,
            )
            out = self.lm_head(pooled)
            return out
        else: #中间stage
            hidden_states= self.llama_model(
            input_ids=None, 
            attention_mask=None,
            inputs_embeds = x,
            )
            return hidden_states
    
if __name__ == '__main__':
    config = PPLlamaConfig()
    initialize_galaxy(config)
    args = get_args()
    config.update_pp_stage_config(args)
    config.print_config()
    tokenizer = BertTokenizer.from_pretrained(config.vocab_path)#TODO: 不能用LlamaTokenizer
    # tokenizer = LlamaTokenizer.from_pretrained( "../../../llama-7b-hf/llama_7b_hf_weight")
    # Prepare Dataset
    start_time = time.time()
    print("Loading data...")
    train_data, dev_data, test_data = build_dataset(config, tokenizer)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)
    
    # Prepare Model
    model = StageModel(config).to(config.device)
    if config.train:
        model.train()
        print('number of llama_model parameters:', get_parameter_number(model.llama_model))
        if config.is_last_stage:
            print('number of lm_head parameters:', get_parameter_number(model.lm_head))
        print("Start training")
    else:
        model.eval()
        print("Start inferencing")
        
    # Prepare PipelineRuntime
    runtime = PipelineRuntime(config, 
                              model, 
                              loss_func=F.cross_entropy, 
                              train_iter=train_iter, 
                              optimizer=torch.optim.SGD, 
                              lr=0.01, 
                              if_cuda=True)
    start_time = time.time()
    #TODO: train_iter 会用完
    for i in range(config.num_iterations):
        runtime.forward_backward_pipelining()
    print("Finish...")
    time_usage = get_time_dif(start_time)
    print(time_usage)
    print(f"{time_usage.seconds} (seconds)")
    get_max_memory(config)