import torch
import torch.nn as nn
def clean_up():
    torch.distributed.destroy_process_group()

def print_rank_0(message):
    """If distributed is initialized, print only on rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)
        
def get_max_memory(config):
    max_memory = torch.cuda.max_memory_allocated(device=config.device)
    print("Max memory:  {} ( {} MB ) ".format( max_memory , max_memory /(1024*1024) ))



