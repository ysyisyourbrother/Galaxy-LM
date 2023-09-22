from distutils.command.config import config
import torch

from galaxy.global_vars import initial_args, get_args

def initialize_galaxy(config):
    initial_args()
    _initialize_distributed(config)


def _initialize_distributed(config):
    args = get_args()

    torch.distributed.init_process_group(
        backend=config.distributed_backend,
        init_method=config.init_method,
        world_size=args.world,
        rank=args.rank,
    )

    # 如果是混合并行，进一步初始化TP、SP和PP的group
    # ...
    