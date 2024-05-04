import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import argparse
from galaxy.data.build import build_dataset, build_iterator,get_time_dif
from galaxy.tokenizer.tokenizer import BertTokenizer
from galaxy.loralib.utils import mark_only_lora_as_trainable, get_parameter_number
from galaxy.utils import get_max_memory
from galaxy.profiler.utils import FlopsProfiler

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',default="bert",type=str)
    parser.add_argument('--config_file',default=None,type=str)
    return parser.parse_args()
args = parse_args()
    
class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.config = config
        if args.model == "bert":
            from galaxy.models.bert.bert_model import BertModel
            self.base_model =  BertModel(config)
        elif  args.model == "bart":
            from galaxy.models.bart.bart_model import BartModel
            self.base_model = BartModel(config)
        elif args.model == "t5":
            from galaxy.models.t5.t5_model import T5Model
            self.base_model = T5Model(config)
        else:
            raise NotImplementedError
        # 最后用一个全连接层将提取到的特征转化为num_class个值
        if hasattr(config, 'hidden_size'):
            self.fc = nn.Linear(config.hidden_size, config.num_classes)
        elif  hasattr(config, 'd_model'):
            self.fc = nn.Linear(config.d_model, config.num_classes)
        else:
            raise NotImplementedError

    def forward(self, x):
        if args.model == "bert":
            # x: (token_ids, seq_len, mask)
            context = (x[0]).to(self.config.device)
            mask = (x[2]).to(self.config.device)
            pooled = self.base_model(context, attention_mask=mask)
            out = self.fc(pooled)
            return out
        else:
            input_ids = (x[0]).to(self.config.device) # [bs,seq]
            sequence_output = self.base_model(
                input_ids=input_ids,
            ) 
            pooled = sequence_output[:,0,:]
            out = self.fc(pooled)
            return out

if __name__ == '__main__':
    if args.model == "bert":
        from train_config.bert.config import BertConfig
        config = BertConfig()
    elif  args.model == "bart":
        from train_config.bart.config import BartConfig
        config = BartConfig()
    elif args.model == "t5":
        from train_config.t5.t5_config import T5Config
        config = T5Config()
    else:
        raise NotImplementedError

    if args.config_file is not None:
        config.load_from_json(args.config_file)
    else:
        print("default config")
    config.print_config()
    
    tokenizer = BertTokenizer.from_pretrained(config.vocab_path)#TODO:  不能用T5 tokenizer
    start_time = time.time()
    print("Loading data...")
    train_data, dev_data, test_data = build_dataset(config, tokenizer)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)

    model = Model(config).to(config.device)
    if config.train:
        model.train()
        print('number of bert parameters:', get_parameter_number(model.base_model))
        print('number of fc parameters:', get_parameter_number(model.fc))
        print("Start training")
    else:
        model.eval()
        print("Start inferencing")
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)
 
    profile_step = 5
    print_profile= True
    prof = FlopsProfiler(model)
    for step in range(10):
        # start profiling at training step "profile_step"
        if step == profile_step:
            prof.start_profile()
        # foward 
        trains, labels = next(train_iter)   
        outputs = model(trains)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()
        # end profiling and print output
        if step == profile_step: # if using multi nodes, check global_rank == 0 as well
            prof.stop_profile()
            flops = prof.get_total_flops()
            macs = prof.get_total_macs()
            params = prof.get_total_params()
            if print_profile:
                prof.print_model_profile(profile_step=profile_step)
            prof.end_profile()
