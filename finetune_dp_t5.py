import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import argparse
from train_config.t5.dp.t5_config import T5Config
from galaxy.tokenizer.tokenizer import BertTokenizer
from galaxy.data.build import build_dataset, build_iterator,get_time_dif
from galaxy.utils import get_max_memory
from torch.nn.parallel import DistributedDataParallel as DDP
from galaxy.initialize import initialize_galaxy
from galaxy.global_vars import get_args
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file',default=None ,type=str)
    return parser.parse_args()

from finetune_t5 import Model

if __name__ == '__main__':
    # Initial Galaxy, args
    config = T5Config()
    initialize_galaxy(config)
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
    mem_after = torch.cuda.memory_allocated()
    print("Model memory usage: {} ( {} MB ) ".format( mem_after-mem_before , (mem_after-mem_before) /(1024*1024) ))
    # print(model)
    # Train
    if config.train:
        model.train()
        print("Start training")
    else:
        model.eval()
        print("Start inferencing")
    # 循环训练
    # Prepare DDP Model 因为每台机器只有一块GPU，所以设备ID始终是0。
    model = DDP(model, device_ids=None,find_unused_parameters=True)
    # model = DDP(model, device_ids=None )
    # TODO: 使用更合适的优化器
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)

    warm_up_iter = 5
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
    allreduce_time_total = 0.0
    run_iter= 10
    
    torch.cuda.synchronize()
    global_start = time.time()
    print("run for  {} iterations".format(run_iter))
    for i in range(run_iter):
        #############################################
        trains, labels = next(train_iter)   
        outputs = model(trains)
        loss = F.cross_entropy(outputs, labels)
        model.zero_grad()
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    global_end = time.time()
    print("global elapse time: {} s for {} iterations".format( global_end - global_start , run_iter))

    get_max_memory(config)