import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import argparse
from train_config.bert.config import BertConfig
from galaxy.data.build import build_dataset, build_iterator,get_time_dif
from galaxy.tokenizer.tokenizer import BertTokenizer
from galaxy.loralib.utils import mark_only_lora_as_trainable, get_parameter_number
from galaxy.utils import get_max_memory
from galaxy.adapters.utils import modify_model_for_peft,get_parameter_number

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file',default=None,type=str)
    return parser.parse_args()
    
class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.config = config
        if config.use_side: # side 
            from galaxy.models.bert.bert_side_model import SideBertModel 
            self.base_model = SideBertModel(config)
        elif config.use_side_only: # forward free side 
            pass
        else:
            from  galaxy.models.bert.bert_model  import BertModel
            self.base_model = BertModel(config)
        modify_model_for_peft(self.base_model, config)
        # 最后用一个全连接层将提取到的特征转化为num_class个值
        self.lm_head = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x):
        # x: (token_ids, seq_len, mask)
        context = (x[0]).to(self.config.device)
        mask = (x[2]).to(self.config.device)
        pooled = self.base_model(context, attention_mask=mask)
        out = self.lm_head(pooled)
        return out


if __name__ == '__main__':
    
    args = parse_args()
    config = BertConfig()
    if args.config_file is not None:
        config.load_from_json(args.config_file)
    else:
        print("default config")
    config.print_config()
    # Prapare Tokenizer
    tokenizer = BertTokenizer.from_pretrained(config.vocab_path)

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
    print(model)
    mem_after = torch.cuda.memory_allocated()
    print("Model memory usage: {} ( {} MB ) ".format( mem_after-mem_before , (mem_after-mem_before) /(1024*1024) ))

    if config.train:
        model.train()
        print('number of bert parameters:', get_parameter_number(model.base_model))
        print('number of lm_head parameters:', get_parameter_number(model.lm_head))
        print("Start training")
    else:
        model.eval()
        print("Start inferencing")
    model.train()
    # TODO: 使用更合适的优化器
    forward_time_total = 0.0
    backward_time_total = 0.0
    start_time = time.time()
    iter = 0
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)
    warm_up_iter =5
    print("warn up for {} iterations".format(warm_up_iter))
    for i in range(warm_up_iter):
        trains, labels = next(train_iter)   
        outputs = model(trains)
        loss = F.cross_entropy(outputs, labels)
        model.zero_grad()
        loss.backward()
        optimizer.step()
    print("warm up finish...")
    
    forward_time_total = 0.0
    backward_time_total = 0.0
    run_iter=  10
    print("run for  {} iterations".format(run_iter))
    torch.cuda.synchronize()
    global_start = time.time()
    for i in range(run_iter):
        #############################################
        trains, labels = next(train_iter)   
        torch.cuda.synchronize()
        start = time.time()
        outputs = model(trains)
        loss = F.cross_entropy(outputs, labels)
        torch.cuda.synchronize()
        end = time.time()
        forward_time_total += (end - start)
        ######################################################
        model.zero_grad()
        loss.backward()
        optimizer.step()
        #########################################################
    torch.cuda.synchronize()
    global_end = time.time()
    backward_time_total = global_end - global_start - forward_time_total
    print("forward time: {} s for {} iterations".format(forward_time_total, run_iter ))
    print("backward time: {} s for {} iterations".format(backward_time_total , run_iter))
    print("global elapse time: {} s for {} iterations".format(global_end - global_start, run_iter ))
    get_max_memory(config)