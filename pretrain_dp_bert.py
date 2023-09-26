import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

import time

from pretrain_config.dp_bert_config import config
from galaxy.data.build import build_dataset, build_iterator,get_time_dif
import galaxy.models.bert.bert_model as bert_model
from galaxy.tokenizer.tokenizer import BertTokenizer
from galaxy.initialize import initialize_galaxy
from galaxy.global_vars import get_args

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = bert_model.BertModel(config)
        for param in self.bert.parameters():
            param.requires_grad = True
        # 最后用一个全连接层将提取到的特征转化为num_class个值
        self.fc = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x):
        # 每一个input的维度：(token_ids, int(label), seq_len, mask)
        context = x[0]
        mask = x[2]
        _, pooled = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        out = self.fc(pooled)
        return out


if __name__ == '__main__':
    # Initial Galaxy, args
    initialize_galaxy(config)
    args = get_args()

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
    model.train()

    # Prepare DDP Model 因为每台机器只有一块GPU，所以设备ID始终是0。
    ddp_model = DDP(model, device_ids=[0])

    # TODO: 使用更合适的优化器
    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=config.learning_rate)
    for i, (trains, labels) in enumerate(train_iter):
        outputs = ddp_model(trains)
        ddp_model.zero_grad()
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        # 这里的操作会触发DDP进行梯度同步
        optimizer.step()
        
        print("finish one iteration.")
        break