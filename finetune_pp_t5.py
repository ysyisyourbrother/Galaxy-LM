import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import argparse
from train_config.t5.t5_pp_config import T5PPConfig
from galaxy.models.t5.t5_pp_model import T5PPModel
from galaxy.tokenizer.tokenizer import BertTokenizer
from galaxy.data.build import build_dataset, build_iterator,get_time_dif
from galaxy.initialize import initialize_galaxy,get_args
from galaxy.loralib.utils import mark_only_lora_as_trainable, get_parameter_number
from galaxy.utils import get_max_memory
from galaxy.core.pipeline_parallel.schedules_t5 import PipelineRuntime


class  StageModel(nn.Module):
    def __init__(self, config):
        super(StageModel, self).__init__()
        self.config = config
        self.base_model = T5PPModel(config)
        if config.is_last_stage: # 最后一个stage，有FC 分类层
            self.lm_head = nn.Linear(config.d_model, config.num_classes, bias=False)
    def forward(self, x):
        if self.config.is_encoder:
            if self.config.is_encoder_first :  # 第一层 x 是input数据
                context = (x[0]).to(self.config.device)
                encoder_outputs = self.base_model(input_ids = context)
                return encoder_outputs
            else: # x只有前面encoder hidden_states
                inputs_embeds = x[0]
                print(inputs_embeds.shape)
                assert inputs_embeds is not None
                encoder_outputs = self.base_model(
                                                    input_ids = None, 
                                                    inputs_embeds = inputs_embeds)
                return encoder_outputs
        else: #decoder x是encoder_outputs和decoder_inputs_embeds
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

            
if __name__ == '__main__':
    config = T5PPConfig()
    initialize_galaxy(config)
    args = get_args()
    config.update_pp_stage_config(args)
    config.print_config()
    tokenizer = BertTokenizer.from_pretrained(config.vocab_path)#TODO: 
    
    start_time = time.time()
    print("Loading data...")
    train_data, dev_data, test_data = build_dataset(config, tokenizer)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    mem_before = torch.cuda.memory_allocated()
    model = StageModel(config).to(config.device)
    mem_after = torch.cuda.memory_allocated()
    print("Model memory usage: {} ( {} MB ) ".format( mem_after-mem_before , (mem_after-mem_before) /(1024*1024) ))
    print(model)
    if config.train:
        model.train()
        print('number of base_model parameters:', get_parameter_number(model.base_model))
        if config.is_last_stage:
            print('number of lm_head parameters:', get_parameter_number(model.lm_head))
        print("Start training")
    else:
        model.eval()
        print("Start inferencing")
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
    time.sleep(30)
    print("Finish...")
    time_usage = get_time_dif(start_time)
    print(time_usage)
    print(f"{time_usage.seconds} (seconds)")
    get_max_memory(config)