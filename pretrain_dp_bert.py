import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import time
from train_config.bert.dp_bert_config import config
from galaxy.data.build import build_dataset, build_iterator,get_time_dif
import galaxy.models.bert.bert_model as bert_model
from galaxy.tokenizer.tokenizer import BertTokenizer
from galaxy.initialize import initialize_galaxy
from galaxy.utils import clean_up
from galaxy.global_vars import get_args
from galaxy.loralib.utils import mark_only_lora_as_trainable, get_parameter_number


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.config = config
        self.bert = bert_model.BertModel(config)
        if not config.use_lora or config.lora_att_dim == 0:
            print("not use lora, train full parameters")
            for param in self.bert.parameters():
                param.requires_grad = True
        else:
            print("use lora...")
            mark_only_lora_as_trainable(self.bert)
        # 最后用一个全连接层将提取到的特征转化为num_class个值
        self.fc = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x):
        
        # x: (token_ids, seq_len, mask)
        context = (x[0]).to(self.config.device)
        mask = (x[2]).to(self.config.device)
        pooled = self.bert(context, attention_mask=mask)
        out = self.fc(pooled)
        return out



if __name__ == '__main__':
    # Initial Galaxy, args
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
        print('number of bert parameters:', get_parameter_number(model.bert)) 
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
    print(f"{time_usage.seconds} (seconds)")
    max_memory = torch.cuda.max_memory_allocated(device=config.device)
    print("Max memory:  {} ( {} MB ) ".format( max_memory , max_memory /(1024*1024) ))
    