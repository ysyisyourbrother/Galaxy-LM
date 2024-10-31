import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import argparse

from  train_config.llama.config import LlamaConfig
from  galaxy.models.llama.llama_model import LlamaModel
from transformers import   LlamaTokenizer
from galaxy.data.build import build_dataset, build_iterator,get_time_dif
from galaxy.loralib.utils import mark_only_lora_as_trainable, get_parameter_number

from galaxy.utils import get_max_memory
from galaxy.adapters.utils import modify_model_for_peft,get_parameter_number

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file',default=None ,type=str)
    return parser.parse_args()
class  Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.config = config
        self.base_model = LlamaModel(config)
        modify_model_for_peft(self.base_model, config)
        self.lm_head = nn.Linear(config.hidden_size, config.num_classes)
    def forward(self, x):
        context = (x[0]).to(self.config.device) # [bs,seq]
        mask = (x[2]).to(self.config.device)# [bs,seq]
        # print(context.shape, mask.shape)
        outputs = self.base_model(
            input_ids=context,
            attention_mask=mask,
        )
        hidden_states = outputs[0]
        pooled_output = hidden_states[:, 0]
        out = self.lm_head(pooled_output)
        return out
    
if __name__ == '__main__':
    args = parse_args()
    config = LlamaConfig()
    if  args.config_file != None:
        config.load_from_json(args.config_file)
    else:
        print("default config")
    config.print_config()
    tokenizer = LlamaTokenizer.from_pretrained('galaxy/models/llama') 
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
    mem_before = torch.cuda.memory_allocated()
    model = Model(config).to(config.device)
    mem_after = torch.cuda.memory_allocated()
    print("Model memory usage: {} ( {} MB ) ".format( mem_after-mem_before , (mem_after-mem_before ) /(1024*1024) ))
    if config.train:
        model.train()
        print('number of llama_model parameters:', get_parameter_number(model.base_model))
        print('number of lm_head parameters:', get_parameter_number(model.lm_head))
        print("Start training")
    else:
        model.eval()
        print("Start inferencing")
    torch.cuda.synchronize()
    start_time = time.time()
    print("seq num: ",len(train_iter))
    print("seq length: ",config.pad_size)
    print("batch size: ",config.batch_size)
    if config.train:
        optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)
        for i in range(config.num_epochs):
            print("epoch: ",i)
            for i, (trains, labels) in enumerate(train_iter):
                outputs = model(trains)    
                model.zero_grad()
                loss = F.cross_entropy(outputs, labels)
                loss.backward() 
                optimizer.step()
    else:
        for i, (trains, labels) in enumerate(train_iter):
            with torch.inference_mode():
                outputs = model(trains)    
    print("Finish...")
    torch.cuda.synchronize()
    end_time = time.time()
    elapsed_time_ms = (end_time - start_time) * 1000
    print("time(ms) = {:.2f}, num of seq = {}, seq_len = {}".format(elapsed_time_ms, len(train_iter), config.pad_size))
    get_max_memory(config)
    