import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import argparse
from train_config.t5.t5_config import T5Config
from galaxy.models.t5.t5_model import T5Model
from galaxy.tokenizer.tokenizer import BertTokenizer
from galaxy.data.build import build_dataset, build_iterator,get_time_dif
from galaxy.loralib.utils import mark_only_lora_as_trainable, get_parameter_number
from galaxy.utils import get_max_memory
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file',default=None ,type=str)
    return parser.parse_args()

class  Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.config = config
        self.base_model = T5Model(config)
        self.lm_head = nn.Linear(config.d_model, config.num_classes, bias=False)
    def forward(self, x):
        input_ids = (x[0]).to(self.config.device) # [bs,seq]
        sequence_output = self.base_model(
            input_ids=input_ids,
        ) 
        pooled = sequence_output[:,0,:]
        out = self.lm_head(pooled)
        return out

if __name__ == '__main__':
    args = parse_args()
    config = T5Config()
    if  args.config_file != None:
        config.load_from_json(args.config_file)
    else:
        print("default config")
    config.print_config()
    tokenizer = BertTokenizer.from_pretrained(config.vocab_path)#TODO:  不能用T5 tokenizer
    start_time = time.time()
    print("Loading data...")
    train_data, dev_data, test_data = build_dataset(config, tokenizer)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)
    
    mem_before = torch.cuda.memory_allocated()
    model = Model(config).to(config.device)
    mem_after = torch.cuda.memory_allocated()
    print("Model memory usage: {} ( {} MB ) ".format( mem_after-mem_before , (mem_after-mem_before) /(1024*1024) ))
    if config.train:
        model.train()
        print('number of base model parameters:', get_parameter_number(model.base_model))
        print('number of fc parameters:', get_parameter_number(model.lm_head))
        print("Start training")
    else:
        model.eval()
        print("Start inferencing")
    start_time = time.time()
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)
    for i in range(config.num_epochs):
        print("epoch: ",i)
        for i, (trains, labels) in enumerate(train_iter):
            outputs = model(trains)
            if config.train:
                optimizer.zero_grad()
                loss = F.cross_entropy(outputs, labels)
                loss.backward() 
                optimizer.step()
    print("Finish...")
    time_usage = get_time_dif(start_time)
    print(time_usage)
    print(f"{time_usage.seconds} (seconds)")
    get_max_memory(config)