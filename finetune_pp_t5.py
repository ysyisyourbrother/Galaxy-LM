import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from train_config.t5.pp.t5_pp_config import T5PPConfig
from galaxy.models.t5.t5_pp_model import StageModel
from  galaxy.models.t5.t5_pp_side_model import SideStageModel
from galaxy.tokenizer.tokenizer import BertTokenizer
from galaxy.data.build import build_dataset, build_iterator,get_time_dif
from galaxy.initialize import initialize_galaxy,get_args
from galaxy.loralib.utils import get_parameter_number
from galaxy.utils import get_max_memory


if __name__ == '__main__':
    config = T5PPConfig()
    initialize_galaxy(config)
    args = get_args()
    config.update_pp_stage_config(args)
    config.print_config()
    tokenizer = BertTokenizer.from_pretrained(config.vocab_path)#TODO: 
    
    start_time = time.time()
    print("Loading data...")
    train_data, dev_data, test_data = build_dataset(config, tokenizer)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    mem_before = torch.cuda.memory_allocated()
    if config.use_side:
        from galaxy.core.pipeline_parallel.schedules_t5_side import PipelineRuntime
        model = SideStageModel(config).to(config.device)
    elif config.use_side_only:
        raise NotImplementedError
    else:
        from galaxy.core.pipeline_parallel.schedules_t5 import PipelineRuntime
        model = StageModel(config).to(config.device)
    mem_after = torch.cuda.memory_allocated()
    print("Model memory usage: {} ( {} MB ) ".format( mem_after-mem_before , (mem_after-mem_before) /(1024*1024) ))
    # print(model)
    if config.train:
        model.train()
        print('number of base_model parameters:', get_parameter_number(model.base_model))
        if config.is_last_stage:
            print('number of lm_head parameters:', get_parameter_number(model.lm_head))
        print("Start training")
    else:
        model.eval()
        print("Start inferencing")
    print(get_parameter_number(model))
    runtime = PipelineRuntime(config, 
                              model, 
                              loss_func=F.cross_entropy, 
                              train_iter=train_iter, 
                              optimizer=torch.optim.SGD, 
                              lr=0.01, 
                              if_cuda=True)
    
 
    #TODO: train_iter 会用完
    warm_up_iter = 5
    runtime.set_record()
    print("warn up for {} iterations".format(warm_up_iter))
    runtime.run_iteration(warm_up_iter)

    run_iter=  10
    runtime.set_record()
    torch.cuda.synchronize()
    global_start = time.time()
    print("run for  {} iterations".format(run_iter))
    runtime.run_iteration(run_iter)
    torch.cuda.synchronize()
    global_end = time.time()

    print("Stage {} Finish...".format(config.stage))
    print( "forward time: {} s for {} iterations".format( runtime.forward_time_total, run_iter ))
    print("backward time: {} s for {} iterations".format( runtime.backward_time_total , run_iter))
    print("global elapse time: {} s for {} iterations".format( global_end - global_start , run_iter))
    get_max_memory(config)  