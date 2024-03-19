import torch
import torch.nn as nn
import torch.nn.functional as F
import time

from galaxy.data.build import build_dataset, build_iterator,get_time_dif
from galaxy.tokenizer.tokenizer import BertTokenizer
from galaxy.initialize import initialize_galaxy,get_args
from galaxy.loralib.utils import  get_parameter_number
from galaxy.core.pipeline_parallel.schedules_side import PipelineRuntime
from train_config.bert.side_pp_bert_config import SidePPBertConfig
from galaxy.models.bert.side_pp_bert_model import SidePPBertModel
from galaxy.models.bert.bert_side_model import mark_only_side_as_trainable 

class StageModel(nn.Module):
    def __init__(self, config):
        super(StageModel, self).__init__()
        self.bert = SidePPBertModel(config)
        self.config = config
        if config.is_last_stage: # 最后一个stage，有FC 分类层
            self.fc = nn.Linear(config.hidden_size, config.num_classes)
        mark_only_side_as_trainable(self.bert)
    def forward(self, x,  side):   
        # x: (token_ids, int(label), seq_len, mask)
        if self.config.is_first_stage: # 第一个stage
            context, mask = x[0].to(self.config.device), x[2].to(self.config.device)
            hidden_states, side_hidden_states = self.bert(input_ids = context,
                                                        hidden_state = None,
                                                        hidden_state_side = None,
                                                        token_type_ids = None,
                                                        attention_mask = mask
                                                        )
            return hidden_states, side_hidden_states
        elif self.config.is_last_stage: #最后一个stage 经过分类层
            input_ids = torch.zeros(self.config.batch_size, self.config.pad_size).long().to(self.config.device)
            pooled =  self.bert(input_ids = input_ids,
                                hidden_state = x,
                                hidden_state_side = side,
                                token_type_ids=None, 
                                attention_mask=None)
            out = self.fc(pooled)
            return out
        else:
            input_ids = torch.zeros(self.config.batch_size, self.config.pad_size).long().to(self.config.device)
            hidden_states, side_hidden_states  = self.bert(input_ids = input_ids,
                                hidden_state = x,
                                hidden_state_side = side,
                                token_type_ids=None, 
                                attention_mask=None)
            return hidden_states, side_hidden_states 
        
if __name__ == '__main__':
    # Initial Galaxy, args
    config = SidePPBertConfig()
    initialize_galaxy(config)
    args = get_args()
    config.update_pp_stage_config(args)
    config.print_config()
    # Prapare Tokenizer
    tokenizer = BertTokenizer.from_pretrained(config.vocab_path)
    
    # Prepare Dataset
    start_time = time.time()
    print("Loading data...")
    train_data, dev_data, test_data = build_dataset(config, tokenizer)
    # next(train_iter) = (x, seq_len, mask), y
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    print(config.device)
    mem_before = torch.cuda.memory_allocated()
    model = StageModel(config).to(config.device)
    mem_after = torch.cuda.memory_allocated()
    print("Model memory usage: {} ( {} MB ) ".format( mem_after-mem_before , (mem_after-mem_before) /(1024*1024) ))
    
    print('number of bert part parameters:  ', get_parameter_number(model.bert)) 
    runtime = PipelineRuntime(config, 
                              model, 
                              loss_func=F.cross_entropy, 
                              train_iter=train_iter, 
                              optimizer=torch.optim.SGD, 
                              lr=0.01, 
                              if_cuda=True)
    torch.cuda.synchronize()
    start = time.time()
    for i in range(config.num_iterations):
        runtime.forward_backward_pipelining()
    torch.cuda.synchronize()
    end = time.time()
    print("total time: {} s".format((end - start) ))
    max_memory = torch.cuda.max_memory_allocated(device=config.device)
    print("Max memory:  {} ( {} MB ) ".format( max_memory , max_memory /(1024*1024) ))
    print("Finish...")