import torch
import torch.nn as nn
import torch.nn.functional as F
import time
# from pretrain_config.pp_bert_config import config
from galaxy.data.build import build_dataset, build_iterator,get_time_dif
import galaxy.models.bert.pp_bert_model as bert_model
from galaxy.tokenizer.tokenizer import BertTokenizer
from galaxy.initialize import initialize_galaxy
from galaxy.utils import clean_up
from galaxy.global_vars import initial_args, get_args
from galaxy.core.pipeline_parallel.schedules import PipelineRuntime
from galaxy.loralib.utils import mark_only_lora_as_trainable, get_parameter_number
from train_config.bert.pp_bert_config import config



class StageModel(nn.Module):
    def __init__(self, config):
        super(StageModel, self).__init__()
        self.bert = bert_model.PPBertModel(config)
        self.config = config
        if config.is_last_stage: # 最后一个stage，有FC 分类层
            self.fc = nn.Linear(config.hidden_size, config.num_classes)

        if not config.use_lora or config.lora_att_dim == 0:
            print("not use lora, train full parameters")
            for param in self.bert.parameters():
                param.requires_grad = True
        else:
            print("use lora, mark_only_lora_as_trainable")
            mark_only_lora_as_trainable(self.bert)
    
    def forward(self, x):
        # x: (token_ids, int(label), seq_len, mask)
        if self.config.is_first_stage: # 第一个stage
            context, mask = x[0], x[2]
            hidden_states = self.bert(context, 
                                        attention_mask=mask, 
                                        )
            return hidden_states
        elif self.config.is_last_stage: #最后一个stage 经过分类层
            input_ids = torch.zeros(self.config.batch_size, self.config.pad_size).long().to(self.config.device)
            pooled = self.bert(input_ids, 
                                encoder_input=x, 
                                 )
            out = self.fc(pooled)
            return out
        else: #中间stage
            input_ids = torch.zeros(self.config.batch_size, self.config.pad_size).long().to(self.config.device)
            hidden_states = self.bert(input_ids, 
                                        encoder_input=x,
                                      )
            return hidden_states

if __name__ == '__main__':
    # Initial Galaxy, args
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

    # Prepare Model
    print(config.device)
    mem_before = torch.cuda.memory_allocated()
    model = StageModel(config).to(config.device)
    mem_after = torch.cuda.memory_allocated()
    print("Model memory usage: {} ( {} MB ) ".format( mem_after-mem_before , (mem_after-mem_before) /(1024*1024) ))
    if config.train:
        model.train()
        print('number of model parameters:  ', get_parameter_number(model)) 
    else:
        raise NotImplementedError("PipelineRuntime only supports train mode.")
    # Prepare PipelineRuntime
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
    print("total time: {} ms".format((end - start) * 1000))
    max_memory = torch.cuda.max_memory_allocated(device=config.device)
    print("Max memory:  {} ( {} MB ) ".format( max_memory , max_memory /(1024*1024) ))
    print("Finish...")