# galaxy
Work in progress LLM framework.

## Bert
**串行**:
 ``` shell
 python pretrain_bert.py
 ```
 ``` shell
 python pretrain_bert.py ----config_file ./pretrain_config/bert_config.json
 ```
 **Tensor Parallel**
+ 参数:pretrain_config/tp_bert_config.py
``` shell
 python pretrain_tp_bert.py --rank 0 --world 2
 python pretrain_tp_bert.py --rank 1 --world 2
```
此时config 划分是默认world_size = 2
可以用参数`--config_file`指定config
``` shell
 python pretrain_tp_bert.py --rank 0 --world 2 --config_file ./pretrain_config/nano_config/tp_bert_config_rank0.json
 python pretrain_tp_bert.py --rank 1 --world 2 --config_file ./pretrain_config/nano_config/tp_bert_config_rank1.json
```
**Data Parallel**:

+ 参数:pretrain_config/dp_bert_config.py
``` shell
 python pretrain_dp_bert.py --rank 0 --world 2
 python pretrain_dp_bert.py --rank 1 --world 2
```
``` shell
 python pretrain_dp_bert.py --rank 0 --world 2 --config_file ./pretrain_config/dp_bert_config.json
 python pretrain_dp_bert.py --rank 1 --world 2 --config_file ./pretrain_config/dp_bert_config.json
```
**Sequence Parallel**
+ 参数:pretrain_config/sp_bert_config.py
``` shell
 python pretrain_sp_bert.py --rank 0 --world 2
 python pretrain_sp_bert.py --rank 1 --world 2
```


**Pipeline**
+ 参数:
  + pretrain_config/pp_bert_config0
  + pretrain_config/pp_bert_config1
``` shell
 python pretrain_pp_bert.py --rank 0 --world 2
 python pretrain_pp_bert.py --rank 1 --world 2
```
**Galaxy**
+ 参数: galaxy_bert_config.py
``` shell
 python pretrain_galaxy.py --rank 0 --world 2
 python pretrain_galaxy.py --rank 1 --world 2
```

## Nano
config:  ./pretrain_config/nano_config/

export GLOO_SOCKET_IFNAME=eth0

rank 0 : 192.168.124.4

``` shell
 python pretrain_tp_bert.py --rank 0 --world 4 --config_file ./pretrain_config/nano_config/tp_bert_config_rank0.json
 python pretrain_tp_bert.py --rank 1 --world 4 --config_file ./pretrain_config/nano_config/tp_bert_config_rank1.json
 python pretrain_tp_bert.py --rank 2 --world 4 --config_file ./pretrain_config/nano_config/tp_bert_config_rank2.json
 python pretrain_tp_bert.py --rank 3 --world 4 --config_file ./pretrain_config/nano_config/tp_bert_config_rank3.json
```

``` shell
 python pretrain_galaxy.py --rank 0 --world 4 --config_file ./pretrain_config/nano_config/galaxy_bert_config_rank0.json
 python pretrain_galaxy.py --rank 1 --world 4 --config_file ./pretrain_config/nano_config/galaxy_bert_config_rank1.json
 python pretrain_galaxy.py --rank 2 --world 4 --config_file ./pretrain_config/nano_config/galaxy_bert_config_rank2.json
 python pretrain_galaxy.py --rank 3 --world 4 --config_file ./pretrain_config/nano_config/galaxy_bert_config_rank3.json
```
### 模型结构
BertLayer结构: ATT -- CON 1-- MLP -- CON 2 
+ 进入每个block之前,数据就准备好了
+ 在block离开的时候，执行通信操作，为下一个block准备数据 （但是不是每个block结束的地方都需要通信）

**SP**:  

X执行 split scatter -->  ATT （中间gather K，V） --> CON -->  MLP  --> CON --> ... --> 最终output： gather along dim <br>

**TP**:  

X执行 copy to all -> ATT 结束 all reduce --> CON --> MLP 结束 all reduce --> CON  --> ATT 结束 all reduce --> CON -->  <br>

**Galaxy**:

X--ATT (TP) -- CON 1 (SP) -- MLP  (TP) -- CON 2 (SP)  <br>
X执行 copy to all -- ATT 结束 reduce scatter --> CON 结束 all gather --> MLP 结束 reduce scatter --> CON 结束 all gather --> ATT .... <br>

**Galaxy2**:

X--ATT (TP) -- CON 1 (SP) -- MLP  (SP) -- CON 2 (SP)  <br>
X copy to all -- ATT 结束 reduce scatter  -- CON -- MLP -->  CON 结束 all gather --> ATT .... <br>


## LoRA 
详细介绍:galaxy/loralib/readme.md

config里lora部分参数：
 ```python
   self.use_lora = True
   self.lora_att_dim = 4
   self.lora_alpha = 32
   self.lora_dropout = 0.1
   self.fan_in_fan_out = True
   self.merge_weights = False
 ```
+ lora_dim:LoRA的秩，lora_r越低代表可训练的参数越少，显存和速度的优化效果更好, 但有可能会损失训练效果
+ lora_alpha: lora训练的时候一般会增加学习率, 系数为 lora_alpha/lora_r	
+ lora_dropout: dropout,避免过拟合

 
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
