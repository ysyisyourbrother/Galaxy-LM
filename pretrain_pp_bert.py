import torch
import torch.nn as nn
import torch.nn.functional as F
import time
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

class StageModel(nn.Module):
    def __init__(self, config):
        super(StageModel, self).__init__()
        self.bert = bert_model.PPBertModel(config)
        self.config = config
        if config.post_process: # 最后一个stage，有FC 分类层
            self.fc = nn.Linear(config.hidden_size, config.num_classes)
 
        if not config.use_lora or config.lora_att_dim == 0:
            print("not use lora, train full parameters")
            for param in self.bert.parameters():
                param.requires_grad = True
        else:
            print("use lora, mark_only_lora_as_trainable")
            mark_only_lora_as_trainable(self.bert)
    
    def forward(self, x):
        # x: (token_ids, int(label), seq_len, mask)
        if config.pre_process: # 第一个stage
            context, mask = x[0], x[2]
            encoded_layers, _ = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
            return encoded_layers
        elif config.post_process: #最后一个stage 经过分类层
            input_ids = torch.zeros(self.config.batch_size, self.config.pad_size).long().to(self.config.device)
            _, pooled = self.bert(input_ids, encoder_input=x, output_all_encoded_layers=False)
            out = self.fc(pooled)
            return out
        else: #中间stage
            input_ids = torch.zeros(self.config.batch_size, self.config.pad_size).long().to(self.config.device)
            encoded_layers, _ = self.bert(input_ids, encoder_input=x, output_all_encoded_layers=False)
            return encoded_layers

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
    # next(train_iter) = (x, seq_len, mask), y
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # Prepare Model
    model = StageModel(config).to(config.device)
    if config.train:
        model.train()
        print('number of model parameters:', get_parameter_number(model)) 
    else:
        model.eval()
        print("Start inferencing")
    # Prepare PipelineRuntime
    runtime = PipelineRuntime(config, 
                              model, 
                              loss_func=F.cross_entropy, 
                              train_iter=train_iter, 
                              optimizer=torch.optim.SGD, 
                              lr=0.01, 
                              if_cuda=True)
    start_time = time.time()
    training_iteration = 4
    for i in range(training_iteration):
        runtime.forward_backward_pipelining()
    print("Finish...")
    time_usage = get_time_dif(start_time)
    print(time_usage)
    print(f"{time_usage.seconds} (seconds)")
    # clean_up() TODO:会Aborted (core dumped)