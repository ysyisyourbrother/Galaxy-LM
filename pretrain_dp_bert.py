import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import time
from train_config.bert.dp.config import  BertConfig
from galaxy.data.build import build_dataset, build_iterator,get_time_dif
from galaxy.tokenizer.tokenizer import BertTokenizer
from galaxy.initialize import initialize_galaxy
from galaxy.utils import clean_up
from galaxy.global_vars import get_args
from galaxy.loralib.utils import mark_only_lora_as_trainable, get_parameter_number
from galaxy.utils import get_max_memory

from pretrain_bert import Model


if __name__ == '__main__':
    # Initial Galaxy, args
    config = BertConfig()
    initialize_galaxy(config)
    args = get_args()
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
    
    # Train
    if config.train:
        model.train()
        print('number of bert parameters:', get_parameter_number(model.base_model)) 
        print('number of fc parameters:', get_parameter_number(model.fc)) 
        print("Start training")
    else:
        model.eval()
        print("Start inferencing")

    # Prepare DDP Model 因为每台机器只有一块GPU，所以设备ID始终是0。
    ddp_model = DDP(model, device_ids=None)
    
    # TODO: 使用更合适的优化器
    start_time = time.time()
    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=config.learning_rate)
    for i in range(config.num_epochs):
        print("epoch: ",i)
        for i, (trains, labels) in enumerate(train_iter):
            outputs = ddp_model(trains)
            if config.train:
                ddp_model.zero_grad()
                loss = F.cross_entropy(outputs, labels)
                loss.backward()
                optimizer.step()
    # clean_up()
    print("Finish...")
    time_usage = get_time_dif(start_time)
    print(time_usage)
    print(f"{time_usage.seconds} (seconds)")
    get_max_memory(config)