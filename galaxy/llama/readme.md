# LLaMA

llama: https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/llama.md

src: https://github.com/huggingface/transformers/tree/main/src/transformers/models/llama
让然后将`modeling_llama.py` `configuration_llama.py`中相对路径修改为绝对路径

```python
# before
from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache, StaticCache
...
# after
from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
...
```

llama-7b-hf 权重: Huggingface 上 https://huggingface.co/luodian/llama-7b-hf/tree/main 下载 (This works perfectly for transformers>=4.28.0.)

```lua
LlamaForCausalLM(
  (model): LlamaModel(
    (embed_tokens): Embedding(32000, 4096, padding_idx=0)
    (layers): ModuleList(
      (0-31): 32 x LlamaDecoderLayer(
        (self_attn): LlamaAttention(
          (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (k_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (v_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (rotary_emb): LlamaRotaryEmbedding()
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear(in_features=4096, out_features=11008, bias=False)
          (down_proj): Linear(in_features=11008, out_features=4096, bias=False)
          (up_proj): Linear(in_features=4096, out_features=11008, bias=False)
          (act_fn): SiLUActivation()
        )
        (input_layernorm): LlamaRMSNorm()
        (post_attention_layernorm): LlamaRMSNorm()
      )
    )
    (norm): LlamaRMSNorm()
  )
  (lm_head): Linear(in_features=4096, out_features=32000, bias=False)
)
```

## 用 Alpaca Fine tuning

从 https://github.com/tatsu-lab/stanford_alpaca
下载

- alpaca_data.json
- prompt.txt

note : 处理数据时间比较久，测试可以从 alpaca_data.json 复制几条建一个小的 json 数据

```bash
python finetune_alpaca.py \
--model_name_or_path ../../../llama-7b-hf/llama_7b_hf_weight \
--data_path ./alpaca/alpaca_data.json \
--bf16 True \
--output_dir output \
--num_train_epochs 1 \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 1 \
--gradient_accumulation_steps 8 \
--evaluation_strategy "no" \
--save_strategy "steps" \
--save_steps 2000 \
--save_total_limit 1 \
--learning_rate 2e-5 \
--weight_decay 0. \
--warmup_ratio 0.03 \
--lr_scheduler_type "cosine" \
--logging_steps 1 \
--tf32 True \
--report_to "none" \
--loggingdir ./log
```

部分参数的意义
https://huggingface.co/docs/transformers/v4.37.1/en/main_classes/trainer#transformers.TrainingArguments

- bf16:
  - bool, optional, defaults to False
  - bf16 16-bit (mixed) precision training
- evaluation_strategy:
  - str or IntervalStrategy, optional, defaults to "no"
  - evaluation strategy to adopt during training
- gradient_accumulation_steps:

  - int, optional, defaults to 1
  - Number of updates steps to accumulate the gradients for, before performing a backward/update pass.
  - logging, evaluation, save will be conducted every gradient_accumulation_steps \* xxx_step training examples

- save_steps
  - int or float, optional, defaults to 500
  - Number of updates steps before two checkpoint saves if save_strategy="steps".
  - Should be an integer or a float in range [0,1). If smaller than 1, will be interpreted as ratio of total training steps.
- save_total_limit

  - int, optional
  - If a value is passed, will limit the total amount of checkpoints.

- logging_steps
  - int or float, optional, defaults to 500
  - Number of update steps between two logs if logging_strategy="steps". Should be an integer or a float in range [0,1). If smaller than 1, will be interpreted as ratio of total training steps.
- report_to
  - str or List[str], optional, defaults to "all"
  - The list of integrations to report the results and logs to. Supported platforms are "azure_ml", "clearml", "codecarbon", "comet_ml", "dagshub", "dvclive", "flyte", "mlflow", "neptune", "tensorboard", and "wandb". Use "all" to report to all integrations installed, "none" for no integrations.
