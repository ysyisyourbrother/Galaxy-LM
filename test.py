import torch
import torch.distributed as dist

import argparse
import time
import distributed_gloo_cpp

parser = argparse.ArgumentParser(description='rank: device id')
parser.add_argument('--rank', default=0, type=int)
parser.add_argument('--world', default=1, type=int)
args = parser.parse_args()

# export GLOO_SOCKET_IFNAME=eth0
default_pg = dist.init_process_group(backend="gloo", init_method='tcp://0.0.0.0:23456', rank=args.rank, world_size=args.world)

# _, default_pg = dist.distributed_c10d._get_default_group()

# 创建一个张量
data = torch.tensor([1, 2, 3, 4])

# 使用all_gather收集数据
gathered_data = [torch.zeros_like(data) for _ in range(dist.get_world_size())]

work = distributed_gloo_cpp.allgather(default_pg, [gathered_data], [data])
work.wait()

# dist.all_gather(gathered_data, data)

# 打印收集到的数据
print(gathered_data)