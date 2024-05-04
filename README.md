# galaxy

Work in progress LLM framework.

## Bert

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

### scriptes

`config`: `train_config/bert`

`model`: `/root/nfs/codespace/Galaxy-LM/galaxy/models/bert`

**串行**:

```shell
python pretrain_bert.py
python pretrain_bert.py --config_file train_config/bert/full.json
python pretrain_bert.py --config_file train_config/bert/lora.json
python pretrain_bert.py --config_file train_config/bert/adapter.json
python pretrain_bert.py --config_file train_config/bert/side.json

```

- full: full model
- lora: LoRA
- adapter: Adapter tuning
- side: side-tuning

**Tensor Parallel**

- 参数:
  - `./train_config/bert/tp/`
  - tp_num_attention_heads_list: attention head 划分
  - tp_intermediate_size_list: 矩阵划分

```shell
python pretrain_tp_bert.py   --world 2 --rank 0
python pretrain_tp_bert.py   --world 2 --rank 1
```

此时 config 划分是默认 world_size = 2
可以用参数`--config_file`指定 config

```shell
python pretrain_tp_bert.py --config_file train_config/bert/tp/full.json --world 2 --rank 0
python pretrain_tp_bert.py --config_file train_config/bert/tp/full.json --world 2 --rank 1
```

**Data Parallel**:

- 参数:
  - `./train_config/bert/dp/`

```shell
python pretrain_dp_bert.py --config_file train_config/bert/dp/full.json --world 2 --rank 0
python pretrain_dp_bert.py --config_file train_config/bert/dp/full.json --world 2 --rank 1

python pretrain_dp_bert.py --config_file train_config/bert/dp/lora.json --world 2 --rank 0
python pretrain_dp_bert.py --config_file train_config/bert/dp/lora.json --world 2 --rank 1

python pretrain_dp_bert.py --config_file train_config/bert/dp/adapter.json --world 2 --rank 0
python pretrain_dp_bert.py --config_file train_config/bert/dp/adapter.json --world 2 --rank 1

python pretrain_dp_bert.py --config_file train_config/bert/dp/side.json --world 2 --rank 0
python pretrain_dp_bert.py --config_file train_config/bert/dp/side.json --world 2 --rank 1
```

**Sequence Parallel**

- 参数:
  - `./train_config/bert/sp/`
  - seq_scatter_list: seq 划分

```shell
python pretrain_sp_bert.py --config_file train_config/bert/sp/full.json --world 2 --rank 0
python pretrain_sp_bert.py --config_file train_config/bert/sp/full.json --world 2 --rank 1

python pretrain_sp_bert.py --config_file train_config/bert/sp/lora.json --world 2 --rank 0
python pretrain_sp_bert.py --config_file train_config/bert/sp/lora.json --world 2 --rank 1
```

**Pipeline**

- 参数:
  - `./train_config/bert/pp/`
  - stage_num_hidden_layers_list: layer 划分

```shell
 python pretrain_pp_bert.py --rank 0 --world 2
 python pretrain_pp_bert.py --rank 1 --world 2
```

```shell
python pretrain_pp_bert.py --config_file train_config/bert/pp/full.json --world 2 --rank 0
python pretrain_pp_bert.py --config_file train_config/bert/pp/full.json --world 2 --rank 1

python pretrain_pp_bert.py --config_file train_config/bert/pp/lora.json --world 2 --rank 0
python pretrain_pp_bert.py --config_file train_config/bert/pp/lora.json --world 2 --rank 1

python pretrain_pp_bert.py --config_file train_config/bert/pp/side.json --world 2 --rank 0
python pretrain_pp_bert.py --config_file train_config/bert/pp/side.json --world 2 --rank 1
```

**Galaxy**

- 参数:
  - `./train_config/bert/galaxy/`
  - con_parallel_method
  - seq_scatter_list
  - att_parallel_method
  - tp_num_attention_heads_list
  - mlp_parallel_method
  - tp_intermediate_size_list

```shell
 python pretrain_galaxy.py --rank 0 --world 2
 python pretrain_galaxy.py --rank 1 --world 2
```

```shell
python pretrain_galaxy.py --config_file train_config/bert/galaxy/full.json --world 2 --rank 0
python pretrain_galaxy.py --config_file train_config/bert/galaxy/full.json --world 2 --rank 1

python pretrain_galaxy.py --config_file train_config/bert/galaxy/lora.json --world 2 --rank 0
python pretrain_galaxy.py --config_file train_config/bert/galaxy/lora.json --world 2 --rank 1
```

## Llama

config: `train_config/llama`
model: `galaxy/models/llama`

```shell
CUDA_VISIBLE_DEVICES=1   python finetune_llama.py --config  train_config/llama/full.json
CUDA_VISIBLE_DEVICES=1   python finetune_llama.py --config  train_config/llama/lora.json
```

**pipeline**

```shell
CUDA_VISIBLE_DEVICES=1   python finetune_pp_llama.py --rank 0 --world 2
 CUDA_VISIBLE_DEVICES=1  python finetune_pp_llama.py --rank 1 --world 2

 CUDA_VISIBLE_DEVICES=1   python finetune_pp_llama.py --rank 0 --world 2 --config  train_config/llama/pp/full.json
 CUDA_VISIBLE_DEVICES=1  python finetune_pp_llama.py --rank 1 --world 2 --config  train_config/llama/pp/full.json
```

## PEFT

### LoRA

src: `galaxy/loralib/`
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

## Adapters

src:`galaxy/adapters/utils.py`

## Profiler

- src: `galaxy/profiler`
- 代码: https://github.com/microsoft/DeepSpeed/tree/master/deepspeed/profiling/flops_profiler
- 用法: https://github.com/microsoft/DeepSpeed/tree/master/deepspeed/profiling/flops_profiler

```shell
python profile_flpos.py --model bert --config  train_config/bert/full.json

python profile_flpos.py  --model bart  --config  train_config/bart/full.json
```
