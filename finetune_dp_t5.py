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
    model = Model(config).to(config.device)
    # print(model)
    # Train
    if config.train:
        model.train()
        print("Start training")
    else:
        model.eval()
        print("Start inferencing")
    # 循环训练
    forward_time_total = 0.0
    backward_time_total = 0.0
    allreduce_time_total = 0.0
    # Prepare DDP Model 因为每台机器只有一块GPU，所以设备ID始终是0。
    ddp_model = DDP(model, device_ids=None,find_unused_parameters=True)
    # TODO: 使用更合适的优化器
    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=config.learning_rate)

    for i in range(config.num_epochs):
        print("epoch: ",i)
        for i, (trains, labels) in enumerate(train_iter):
            ###############################################
            # Forward 
            torch.cuda.synchronize()
            start = time.time()
            outputs = ddp_model(trains)
            loss = F.cross_entropy(outputs, labels)
            torch.cuda.synchronize()
            end = time.time()
            forward_time_total += (end - start)
            ################################################
            # Backward
            torch.cuda.synchronize()
            start = time.time()
            ddp_model.zero_grad()
            loss.backward()
            torch.cuda.synchronize()
            end = time.time()
            backward_time_total += (end - start)
            ##########################################
            torch.cuda.synchronize()
            start = time.time()
            optimizer.step()
            torch.cuda.synchronize()
            end = time.time()
            allreduce_time_total += (end - start)
    # clean_up()
    print("Finish...")
    print("Forward time: {} s".format(forward_time_total))
    print("Backward time: {} s".format(backward_time_total))
    print("Allreduce time: {} s".format(allreduce_time_total))
    get_max_memory(config)