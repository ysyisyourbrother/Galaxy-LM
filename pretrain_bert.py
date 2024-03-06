import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import argparse
from train_config.bert.bert_config import config
from galaxy.data.build import build_dataset, build_iterator,get_time_dif
import galaxy.models.bert.bert_model as bert_model
from galaxy.tokenizer.tokenizer import BertTokenizer
from galaxy.loralib.utils import mark_only_lora_as_trainable, get_parameter_number


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file',default=None,type=str)
    return parser.parse_args()
    
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
            print("use lora")
            mark_only_lora_as_trainable(self.bert)
        # 最后用一个全连接层将提取到的特征转化为num_class个值
        self.fc = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x):
        
        # x: (token_ids, seq_len, mask)
        context = (x[0]).to(self.config.device)
        mask = (x[2]).to(self.config.device)
        _, pooled = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        out = self.fc(pooled)
        return out


if __name__ == '__main__':
    
    args = parse_args()
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
    model = Model(config).to(config.device)

    if config.train:
        model.train()
        print('number of bert parameters:', get_parameter_number(model.bert))
        print('number of fc parameters:', get_parameter_number(model.fc))
        print("Start training")
    else:
        model.eval()
        print("Start inferencing")
        
    # TODO: 使用更合适的优化器
    start_time = time.time()
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)
    for i in range(config.num_epochs):
        print("epoch: ",i)
        for i, (trains, labels) in enumerate(train_iter):
            outputs = model(trains)
            if config.train:
                model.zero_grad()
                loss = F.cross_entropy(outputs, labels)
                loss.backward()
                optimizer.step()
                # print(f"finish {i} iteration.")
        # break
    print("Finish...")
    time_usage = get_time_dif(start_time)
    print(time_usage)
    print(f"{time_usage.seconds} (seconds)")
    