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

if __name__ == '__main__':
    args = parse_args()
    if args.model == "bert":
        from train_config.bert.config import BertConfig
        from pretrain_bert import Model
        config = BertConfig()
    elif  args.model == "bart":
        from train_config.bart.config import BartConfig
        from finetune_bart import Model
        config = BartConfig()
    elif args.model == "t5":
        from train_config.t5.t5_config import T5Config
        from finetune_t5 import Model
        config = T5Config()
    else:
        raise NotImplementedError

    if args.config_file is not None:
        config.load_from_json(args.config_file)
    else:
        print("default config")
    config.print_config()
    
    tokenizer = BertTokenizer.from_pretrained(config.vocab_path) 
    start_time = time.time()
    print("Loading data...")
    train_data, dev_data, test_data = build_dataset(config, tokenizer)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    # model memory
    mem_before = torch.cuda.memory_allocated()
    model = Model(config).to(config.device)
    mem_after = torch.cuda.memory_allocated()
    print("Model memory usage: {} ( {} MB ) ".format( mem_after-mem_before , (mem_after-mem_before) /(1024*1024) ))
    # number of parameters
    print("========= number of parameters ==========")
    total_params = get_parameter_number(model)["Total"]
    trainale_params = get_parameter_number(model)["Trainable"]
    print(f"Total number of parameters: {total_params}")
    print(f"Trainable number of parameters: {trainale_params}")
    print(f"Estimated model size: {total_params * 4 / (1024 ** 2)} MB = {total_params * 4 / (1024 ** 2) / 1024 } GB" )
    

    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)

    # profile memory
    if  config.train == False:
        print("========= Start inferencing =========")
        model.eval()
        with torch.no_grad():
            mem_before = torch.cuda.memory_allocated()
            trains, labels = next(train_iter) 
            outputs = model(trains)
            mem_after = torch.cuda.memory_allocated()
            print("Forward memory usage: {} ( {} MB )  ({} GB )".format( mem_after-mem_before , (mem_after-mem_before) /(1024*1024)  , (mem_after-mem_before) /(1024*1024*1024) ))
            print( "Max Memory {} MB ({} GB ) ".format(torch.cuda.max_memory_allocated( config.device)/(1024*1024), (torch.cuda.max_memory_allocated( config.device)/(1024*1024*1024)) ))

    else:
        print("========= Start training =========")
        gradient_memory = trainale_params * 4 / (1024 ** 2)  # float32 storage
        print(f"Estimated gradient memory usage: {gradient_memory} MB , {gradient_memory / 1024} GB")
    
        for i in range (3):
            optimizer.zero_grad()
            mem_before = torch.cuda.memory_allocated()
            trains, labels = next(train_iter) 
            outputs = model(trains)
            loss = F.cross_entropy(outputs, labels)
            mem_after = torch.cuda.memory_allocated()
            print("Forward memory usage: {} ( {} MB )  ({} GB )".format( mem_after-mem_before , (mem_after-mem_before) /(1024*1024)  , (mem_after-mem_before) /(1024*1024*1024) ))
            mem_before = torch.cuda.memory_allocated()
            loss.backward()
            optimizer.step()
            mem_after = torch.cuda.memory_allocated()
            print("Backward memory usage: {} ( {} MB )  ({} GB )".format( mem_after-mem_before , (mem_after-mem_before) /(1024*1024), (mem_after-mem_before) /(1024*1024*1024) ))
            print( "Max Memory {} MB ({} GB ) ".format(torch.cuda.max_memory_allocated( config.device)/(1024*1024), (torch.cuda.max_memory_allocated( config.device)/(1024*1024*1024)) ))
        
        # profile flops
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
            # end profiling and print output
            if step == profile_step: # if using multi nodes, check global_rank == 0 as well
                prof.stop_profile()
                flops = prof.get_total_flops()
                macs = prof.get_total_macs()
                params = prof.get_total_params()
                if print_profile:
                    prof.print_model_profile(profile_step=profile_step)
                prof.end_profile()
            loss.backward()
            optimizer.step()
