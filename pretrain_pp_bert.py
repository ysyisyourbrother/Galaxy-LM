import torch
import torch.nn as nn
import torch.nn.functional as F
import time
# from pretrain_config.pp_bert_config import config
from galaxy.data.build import build_dataset, build_iterator,get_time_dif

from galaxy.tokenizer.tokenizer import BertTokenizer
from galaxy.initialize import initialize_galaxy
from galaxy.utils import clean_up
from galaxy.global_vars import initial_args, get_args
from galaxy.loralib.utils import mark_only_lora_as_trainable, get_parameter_number
from train_config.bert.pp.config import BertConfig
from galaxy.utils import get_max_memory

from galaxy.adapters.utils import modify_model_for_peft,get_parameter_number




if __name__ == '__main__':
    # Initial Galaxy, args
    config = BertConfig()
    initialize_galaxy(config)
    args = get_args()
    config.update_pp_stage_config(args)
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
    print(config.device)
    mem_before = torch.cuda.memory_allocated()
    if config.use_side:
        from galaxy.models.bert.side_pp_bert_model import StageModel
    else:
        from galaxy.models.bert.pp_bert_model import  StageModel
    model = StageModel(config).to(config.device)
    mem_after = torch.cuda.memory_allocated()
    print("Model memory usage: {} ( {} MB ) ".format( mem_after-mem_before , (mem_after-mem_before) /(1024*1024) ))
    if config.train:
        model.train()
        print('number of model parameters:  ', get_parameter_number(model)) 
    else:
        raise NotImplementedError("PipelineRuntime only supports train mode.")
    # Prepare PipelineRuntime
    if config.use_side:
        from galaxy.core.pipeline_parallel.schedules_side import PipelineRuntime
    else:
        from galaxy.core.pipeline_parallel.schedules import PipelineRuntime

    runtime = PipelineRuntime(config, 
                              model, 
                              loss_func=F.cross_entropy, 
                              train_iter=train_iter, 
                              optimizer=torch.optim.SGD, 
                              lr=0.01, 
                              if_cuda=True)
    torch.cuda.synchronize()
    start = time.time()
    for i in range(config.num_iterations):
        runtime.forward_backward_pipelining()
    torch.cuda.synchronize()
    end = time.time()
    print("total time: {} s".format((end - start) ))
    get_max_memory(config)
    print("Finish...")