# galaxy

Work in progress LLM framework.

## Bert

**串行**:

```shell
python pretrain_bert.py
```

```shell
python pretrain_bert.py --config_file ./pretrain_config/bert_config.json
```

**Tensor Parallel**

- 参数:pretrain_config/tp_bert_config.py

```shell
 python pretrain_tp_bert.py --rank 0 --world 2
 python pretrain_tp_bert.py --rank 1 --world 2
```

此时 config 划分是默认 world_size = 2
可以用参数`--config_file`指定 config

```shell
 python pretrain_tp_bert.py --rank 0 --world 2 --config_file ./pretrain_config/tp_bert_config.json
 python pretrain_tp_bert.py --rank 1 --world 2 --config_file ./pretrain_config/tp_bert_config.json
```

**Data Parallel**:

- 参数:pretrain_config/dp_bert_config.py

```shell
 python pretrain_dp_bert.py --rank 0 --world 2
 python pretrain_dp_bert.py --rank 1 --world 2
```

```shell
 python pretrain_dp_bert.py --rank 0 --world 2 --config_file ./pretrain_config/dp_bert_config.json
 python pretrain_dp_bert.py --rank 1 --world 2 --config_file ./pretrain_config/dp_bert_config.json
```

**Sequence Parallel**

- 参数:pretrain_config/sp_bert_config.py

```shell
 python pretrain_sp_bert.py --rank 0 --world 2
 python pretrain_sp_bert.py --rank 1 --world 2
```

```shell
 python pretrain_sp_bert.py --rank 0 --world 2 --config_file ./pretrain_config/sp_bert_config.json
 python pretrain_sp_bert.py --rank 1 --world 2 --config_file ./pretrain_config/sp_bert_config.json
```

**Pipeline**

- 参数:
  - pretrain_config/pp_bert_config0
  - pretrain_config/pp_bert_config1

```shell
 python pretrain_pp_bert.py --rank 0 --world 2
 python pretrain_pp_bert.py --rank 1 --world 2
```

```shell
 python pretrain_pp_bert.py --rank 0 --world 2 --config_file ./pretrain_config/pp_bert_config0.json
 python pretrain_pp_bert.py --rank 1 --world 2 --config_file ./pretrain_config/pp_bert_config1.json
```

**Galaxy**

- 参数: galaxy_bert_config.py

```shell
 python pretrain_galaxy.py --rank 0 --world 2
 python pretrain_galaxy.py --rank 1 --world 2
```

```shell
 python pretrain_galaxy.py --rank 0 --world 2  --config_file ./pretrain_config/galaxy_bert_config.json
 python pretrain_galaxy.py --rank 1 --world 2  --config_file ./pretrain_config/galaxy_bert_config.json
```

## Nano

config 路径: `./pretrain_config/nano_config/`

通信设置 : `export GLOO_SOCKET_IFNAME=eth0`

rank 0 : 192.168.124.4

### TP

| batch_size | seq_len | epoch | time(seconds) |
| :--------- | :-----: | :---: | :-----------: |
| 10         |   32    |   5   |      18       |

```shell
 python pretrain_tp_bert.py --rank 0 --world 4 --config_file ./pretrain_config/nano_config/tp_bert_config_rank0.json
 python pretrain_tp_bert.py --rank 1 --world 4 --config_file ./pretrain_config/nano_config/tp_bert_config_rank1.json
 python pretrain_tp_bert.py --rank 2 --world 4 --config_file ./pretrain_config/nano_config/tp_bert_config_rank2.json
 python pretrain_tp_bert.py --rank 3 --world 4 --config_file ./pretrain_config/nano_config/tp_bert_config_rank3.json
```

```shell
 python pretrain_galaxy.py --rank 0 --world 4 --config_file ./pretrain_config/nano_config/galaxy_bert_config_rank0.json
 python pretrain_galaxy.py --rank 1 --world 4 --config_file ./pretrain_config/nano_config/galaxy_bert_config_rank1.json
 python pretrain_galaxy.py --rank 2 --world 4 --config_file ./pretrain_config/nano_config/galaxy_bert_config_rank2.json
 python pretrain_galaxy.py --rank 3 --world 4 --config_file ./pretrain_config/nano_config/galaxy_bert_config_rank3.json
```

TODO:
start forward of microbatch 1
Exception in thread Thread-3:
Traceback (most recent call last):
File "/home/brandonye/miniforge3/envs/py36/lib/python3.6/threading.py", line 916, in \_bootstrap_inner
self.run()
File "/home/brandonye/miniforge3/envs/py36/lib/python3.6/threading.py", line 864, in run
self.\_target(\*self.\_args, \*\*self.\_kwargs)
File "/home/brandonye/CodeSpace/Galaxy-LM/galaxy/core/pipeline_parallel/communication.py", line 104, in recv_helper_thread
tensor = \_recv(tensor_shape, src_rank, tag)
File "/home/brandonye/CodeSpace/Galaxy-LM/galaxy/core/pipeline_parallel/communication.py", line 128, in \_recv
dist.recv(tensor, src=src_rank, tag=tag)
File "/home/brandonye/miniforge3/envs/py36/lib/python3.6/site-packages/torch/distributed/distributed_c10d.py", line 976, in recv
pg.recv([tensor], src, tag).wait()
RuntimeError: [/media/nvidia/NVME/pytorch/pytorch-v1.10.0/third_party/gloo/gloo/transport/tcp/pair.cc:598] Connection closed by peer [192.168.124.7]:64397

Exception in thread Thread-5:
Traceback (most recent call last):
File "/home/brandonye/miniforge3/envs/py36/lib/python3.6/threading.py", line 916, in \_bootstrap_inner
self.run()
File "/home/brandonye/miniforge3/envs/py36/lib/python3.6/threading.py", line 864, in run
self.\_target(\*self.\_args, \*\*self.\_kwargs)
File "/home/brandonye/CodeSpace/Galaxy-LM/galaxy/core/pipeline_parallel/communication.py", line 104, in recv_helper_thread
tensor = \_recv(tensor_shape, src_rank, tag)
File "/home/brandonye/CodeSpace/Galaxy-LM/galaxy/core/pipeline_parallel/communication.py", line 128, in \_recv
dist.recv(tensor, src=src_rank, tag=tag)
File "/home/brandonye/miniforge3/envs/py36/lib/python3.6/site-packages/torch/distributed/distributed_c10d.py", line 976, in recv
pg.recv([tensor], src, tag).wait()
RuntimeError: [/media/nvidia/NVME/pytorch/pytorch-v1.10.0/third_party/gloo/gloo/transport/tcp/pair.cc:598] Connection closed by peer [192.168.124.17]:52654

```shell
 python pretrain_pp_bert.py --rank 0 --world 4 --config_file ./pretrain_config/nano_config/pp_bert_config_rank0.json
 python pretrain_pp_bert.py --rank 1 --world 4 --config_file ./pretrain_config/nano_config/pp_bert_config_rank1.json
 python pretrain_pp_bert.py --rank 2 --world 4 --config_file ./pretrain_config/nano_config/pp_bert_config_rank2.json
 python pretrain_pp_bert.py --rank 3 --world 4 --config_file ./pretrain_config/nano_config/pp_bert_config_rank3.json
```

### 模型结构

BertLayer 结构: ATT -- CON 1-- MLP -- CON 2

- 进入每个 block 之前,数据就准备好了
- 在 block 离开的时候，执行通信操作，为下一个 block 准备数据 （但是不是每个 block 结束的地方都需要通信）

**SP**:

X 执行 split scatter --> ATT （中间 gather K，V） --> CON --> MLP --> CON --> ... --> 最终 output： gather along dim <br>

**TP**:

X 执行 copy to all -> ATT 结束 all reduce --> CON --> MLP 结束 all reduce --> CON --> ATT 结束 all reduce --> CON --> <br>

**Galaxy**:

X--ATT (TP) -- CON 1 (SP) -- MLP (TP) -- CON 2 (SP) <br>
X 执行 copy to all -- ATT 结束 reduce scatter --> CON 结束 all gather --> MLP 结束 reduce scatter --> CON 结束 all gather --> ATT .... <br>

**Galaxy2**:

X--ATT (TP) -- CON 1 (SP) -- MLP (SP) -- CON 2 (SP) <br>
X copy to all -- ATT 结束 reduce scatter -- CON -- MLP --> CON 结束 all gather --> ATT .... <br>

## LoRA

详细介绍:galaxy/loralib/readme.md

config 里 lora 部分参数：

```python
  self.use_lora = True
  self.lora_att_dim = 4
  self.lora_alpha = 32
  self.lora_dropout = 0.1
  self.fan_in_fan_out = True
  self.merge_weights = False
```

- lora_dim:LoRA 的秩，lora_r 越低代表可训练的参数越少，显存和速度的优化效果更好, 但有可能会损失训练效果
- lora_alpha: lora 训练的时候一般会增加学习率, 系数为 lora_alpha/lora_r
- lora_dropout: dropout,避免过拟合

```python
# ===== Before =====
# layer = nn.Linear(in_features, out_features)

# ===== After ======
from galaxy.loralib.layers import  Linear as LoraLinear
# Add a pair of low-rank adaptation matrices with rank r=16
layer = LoraLinear(in_features, out_features, r=16)
```

Before the training loop begins, mark only LoRA parameters as trainable.

```python
from galaxy.loralib.utils import mark_only_lora_as_trainable
model = BigModel()
# This sets requires_grad to False for all parameters without the string "lora_" in their names
mark_only_lora_as_trainable(model)
# Training loop
for batch in dataloader:
   ...
```
