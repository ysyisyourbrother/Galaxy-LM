# galaxy
Work in progress LLM framework.


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
