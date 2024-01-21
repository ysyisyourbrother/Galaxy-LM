import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

import time
import importlib

# from pretrain_config.pp_bert_config import config
from galaxy.data.build import build_dataset, build_iterator,get_time_dif
import galaxy.models.bert.pp_bert_model as bert_model
from galaxy.tokenizer.tokenizer import BertTokenizer
from galaxy.initialize import initialize_galaxy
from galaxy.utils import clean_up
from galaxy.global_vars import initial_args, get_args
from galaxy.core.pipeline_parallel.schedules import PipelineRuntime
from galaxy.loralib.utils import mark_only_lora_as_trainable, get_parameter_number


initial_args()
args = get_args()
if args.rank == 0:
    from pretrain_config.pp_bert_config0 import config
else:
    from pretrain_config.pp_bert_config1 import config

class StageModel0(nn.Module):
    def __init__(self, config):
        super(StageModel0, self).__init__()
        self.bert = bert_model.PPBertModel(config)
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
        # x: (token_ids, int(label), seq_len, mask)
        context, mask = x[0], x[2]
        encoded_layers, _ = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        return encoded_layers

class StageModel1(nn.Module):
    def __init__(self, config):
        super(StageModel1, self).__init__()
        self.config = config
        self.bert = bert_model.PPBertModel(config)
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
        # x: (token_ids, int(label), seq_len, mask)
        # TODO: 如何将inputs id 或attention mask从第一个stage传递到后面？
        input_ids = torch.zeros(self.config.batch_size, self.config.pad_size).long().to(self.config.device)
        _, pooled = self.bert(input_ids, encoder_input=x, output_all_encoded_layers=False)
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
    # next(train_iter) = (x, seq_len, mask), y
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # Prepare Model
    if args.rank == 0:
        Model = StageModel0
    elif args.rank == 1:
        Model = StageModel1
    
    model = Model(config).to(config.device)
    model.train()
    print('number of bert parameters:', get_parameter_number(model.bert)) 
    print('number of fc parameters:', get_parameter_number(model.fc)) 
    # Prepare PipelineRuntime
    runtime = PipelineRuntime(config, 
                              model, 
                              loss_func=F.cross_entropy, 
                              train_iter=train_iter, 
                              optimizer=torch.optim.SGD, 
                              lr=0.01, 
                              if_cuda=True)

    training_iteration = 1
    for i in range(training_iteration):
        runtime.forward_backward_pipelining()
    clean_up()