import torch
import torch.nn as nn
import torch.nn.functional as F
import time

from pretrain_config.lora_bert_config import config
from galaxy.data.build import build_dataset, build_iterator,get_time_dif
import galaxy.models.bert.bert_model as bert_model
from galaxy.tokenizer.tokenizer import BertTokenizer
from galaxy.loralib.utils import mark_only_lora_as_trainable, get_parameter_number




class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = bert_model.BertModel(config)
        if  hasattr(config, 'use_lora') and config.lora_att_dim > 0:
            print('use lora')
            mark_only_lora_as_trainable(self.bert)
        # 最后用一个全连接层将提取到的特征转化为num_class个值
        self.fc = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x):
        # x: (token_ids, seq_len, mask)
        context = (x[0]).to(config.device)
        mask = (x[2]).to(config.device)
        _, pooled = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        out = self.fc(pooled)
        return out


if __name__ == '__main__':
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
    model.train()
    # 147456 = 768*4*2*2*12 (hidden_size*lora_att_dim*2(A/B)*2(QV)*num_hidden_layers
    print('number of bert parameters:', get_parameter_number(model.bert))
    # TODO: 使用更合适的优化器
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)
    for i, (trains, labels) in enumerate(train_iter):
        outputs = model(trains)
        model.zero_grad()
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()
        
        print("finish one iteration.")
        break