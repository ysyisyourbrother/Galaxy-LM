

#  llama-moe  
模型：https://huggingface.co/llama-moe/LLaMA-MoE-v1-3_5B-2_8

This model is NOT fine-tuned by instruction pairs, so it may not be good enough to act like a chatbot.
  
num_hidden_layers = 32

``` lua
LlamaMoEForCausalLM(
  (model): LlamaMoEModel(
    (embed_tokens): Embedding(32000, 4096, padding_idx=0)
    (layers): ModuleList(
      (0-31): 32 x LlamaMoEDecoderLayer(
        (self_attn): LlamaAttention(
          (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (k_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (v_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (rotary_emb): LlamaRotaryEmbedding()
        )
        (input_layernorm): LlamaRMSNorm()
        (post_attention_layernorm): LlamaRMSNorm()
        (mlp): LinearGLUMoELayer(
          (gate): TopKBalancedNoisyGate(
            (gate_network): Sequential(
              (0): Linear(in_features=4096, out_features=8, bias=False)
              (1): Tanh()
              (2): Linear(in_features=8, out_features=8, bias=False)
            )
            (softmax): Softmax(dim=1)
            (weight_noise): Linear(in_features=4096, out_features=8, bias=False)
            (softplus): Softplus(beta=1, threshold=20)
          )
          (calculator): UniversalCalculator(
            (experts): LinearGLUExperts(
              in_features=4096, hidden_features=11008, out_features=4096, hidden_act=silu, num_experts=8, size_experts=[1376, 1376, 1376, 1376, 1376, 1376, 1376, 1376], bias=False
              (act_fn): SiLUActivation()
              (weight_gate): ParameterList(
                  (0): Parameter containing: [torch.bfloat16 of size 1376x4096 (GPU 0)]
                  (1): Parameter containing: [torch.bfloat16 of size 1376x4096 (GPU 0)]
                  (2): Parameter containing: [torch.bfloat16 of size 1376x4096 (GPU 0)]
                  (3): Parameter containing: [torch.bfloat16 of size 1376x4096 (GPU 0)]
                  (4): Parameter containing: [torch.bfloat16 of size 1376x4096 (GPU 0)]
                  (5): Parameter containing: [torch.bfloat16 of size 1376x4096 (GPU 0)]
                  (6): Parameter containing: [torch.bfloat16 of size 1376x4096 (GPU 0)]
                  (7): Parameter containing: [torch.bfloat16 of size 1376x4096 (GPU 0)]
              )
              (weight_up): ParameterList(
                  (0): Parameter containing: [torch.bfloat16 of size 1376x4096 (GPU 0)]
                  (1): Parameter containing: [torch.bfloat16 of size 1376x4096 (GPU 0)]
                  (2): Parameter containing: [torch.bfloat16 of size 1376x4096 (GPU 0)]
                  (3): Parameter containing: [torch.bfloat16 of size 1376x4096 (GPU 0)]
                  (4): Parameter containing: [torch.bfloat16 of size 1376x4096 (GPU 0)]
                  (5): Parameter containing: [torch.bfloat16 of size 1376x4096 (GPU 0)]
                  (6): Parameter containing: [torch.bfloat16 of size 1376x4096 (GPU 0)]
                  (7): Parameter containing: [torch.bfloat16 of size 1376x4096 (GPU 0)]
              )
              (weight_down): ParameterList(
                  (0): Parameter containing: [torch.bfloat16 of size 4096x1376 (GPU 0)]
                  (1): Parameter containing: [torch.bfloat16 of size 4096x1376 (GPU 0)]
                  (2): Parameter containing: [torch.bfloat16 of size 4096x1376 (GPU 0)]
                  (3): Parameter containing: [torch.bfloat16 of size 4096x1376 (GPU 0)]
                  (4): Parameter containing: [torch.bfloat16 of size 4096x1376 (GPU 0)]
                  (5): Parameter containing: [torch.bfloat16 of size 4096x1376 (GPU 0)]
                  (6): Parameter containing: [torch.bfloat16 of size 4096x1376 (GPU 0)]
                  (7): Parameter containing: [torch.bfloat16 of size 4096x1376 (GPU 0)]
              )
            )
          )
        )
      )
    )
    (norm): LlamaRMSNorm()
  )
  (lm_head): Linear(in_features=4096, out_features=32000, bias=False)
)

```



 

BaseMoELayer 计算流程
+ 输入经过gate:gate_outputs: dict = self.gate(x)
  ```
  {
   "topK_indices": top_k_indices,
              "topK_scores": top_k_scores,
              "balance_loss": balance_loss,
              "load": load,
              "importance": importance,
          }
  ```

+ 根据gate打分计算MLP结果 calc_outs: CalculatorOutput = self.calculator(x, **gate_outputs)
这里gate_outputs是一个字典
元素: topK_indices topK_scores balance_loss

TopKBalancedNoisyGate: 
``` python
#  [seq_len, hidden_size] -> [seq_len, expert_num]
logits_gate = self.gate_network(x)
```

## Alpaca
Stanford Alpaca：A Strong，Replicable Instruction-Following Model ：
- 基于 LLaMA-7B，使用self-instruct指令调优
-  https://github.com/tatsu-lab/stanford_alpaca  
- 开源
  - 数据集: alpaca_data.json，52k指令数据
  - 生成数据集的代码:  generate_instruction.py，需要 OPENAI_API_KEY（使用openai api生成52k不重复的指令和对应输出）
  - 微调代码（基于huggingface transformer）
  - Alpaca-7B 权重

``` bash
python finetune_alpaca.py \
--model_name_or_path ./llama_moe \
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
--report_to "none"
```

## llama_moe_predict
在llama_moe基础上修改
+ 模型: LlamaMoEForCausalLMPredict 每一侧层增加一个self.predict_gate: TopKBalancedNoisyGate （实际上最后一层可以不需要）
+ outputs:增加返回 all_gate_inputs all_gate_outputs 
finetune代码修改:
+ CustomTrainer继承 transformers.Trainer
  + 重写compute_loss
  + 重写save_model

TODO:
+ predict_gate 初始化方式


``` bash
python finetune_predict.py \
--model_name_or_path ./llama_moe \
--data_path ./alpaca/mini_data.json \
--bf16 True \
--output_dir predict_output \
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
--logging_strategy  "steps" \
--logging_steps 1 \
--tf32 True \
--report_to "none"
```
```
LlamaMoEForCausalLMPredict(
  (model): LlamaMoEModelPredict(
    (embed_tokens): Embedding(32000, 4096, padding_idx=0)
    (layers): ModuleList(
      (0-31): 32 x LlamaMoEDecoderLayer(
        (self_attn): LlamaAttention(
          (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (k_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (v_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (rotary_emb): LlamaRotaryEmbedding()
        )
        (input_layernorm): LlamaRMSNorm()
        (post_attention_layernorm): LlamaRMSNorm()
        (mlp): LinearGLUMoELayer(
          (gate): TopKBalancedNoisyGate(
            (gate_network): Sequential(
              (0): Linear(in_features=4096, out_features=8, bias=False)
              (1): Tanh()
              (2): Linear(in_features=8, out_features=8, bias=False)
            )
            (softmax): Softmax(dim=1)
            (weight_noise): Linear(in_features=4096, out_features=8, bias=False)
            (softplus): Softplus(beta=1, threshold=20)
          )
          (predict_gate): TopKBalancedNoisyGate(
            (gate_network): Sequential(
              (0): Linear(in_features=4096, out_features=8, bias=False)
              (1): Tanh()
              (2): Linear(in_features=8, out_features=8, bias=False)
            )
            (softmax): Softmax(dim=1)
            (weight_noise): Linear(in_features=4096, out_features=8, bias=False)
            (softplus): Softplus(beta=1, threshold=20)
          )
          (calculator): UniversalCalculator(
            (experts): LinearGLUExperts(
              in_features=4096, hidden_features=11008, out_features=4096, hidden_act=silu, num_experts=8, size_experts=[1376, 1376, 1376, 1376, 1376, 1376, 1376, 1376], bias=False
              (act_fn): SiLUActivation()
              (weight_gate): ParameterList(
                  (0): Parameter containing: [torch.bfloat16 of size 1376x4096 (GPU 0)]
                  (1): Parameter containing: [torch.bfloat16 of size 1376x4096 (GPU 0)]
                  (2): Parameter containing: [torch.bfloat16 of size 1376x4096 (GPU 0)]
                  (3): Parameter containing: [torch.bfloat16 of size 1376x4096 (GPU 0)]
                  (4): Parameter containing: [torch.bfloat16 of size 1376x4096 (GPU 0)]
                  (5): Parameter containing: [torch.bfloat16 of size 1376x4096 (GPU 0)]
                  (6): Parameter containing: [torch.bfloat16 of size 1376x4096 (GPU 0)]
                  (7): Parameter containing: [torch.bfloat16 of size 1376x4096 (GPU 0)]
              )
              (weight_up): ParameterList(
                  (0): Parameter containing: [torch.bfloat16 of size 1376x4096 (GPU 0)]
                  (1): Parameter containing: [torch.bfloat16 of size 1376x4096 (GPU 0)]
                  (2): Parameter containing: [torch.bfloat16 of size 1376x4096 (GPU 0)]
                  (3): Parameter containing: [torch.bfloat16 of size 1376x4096 (GPU 0)]
                  (4): Parameter containing: [torch.bfloat16 of size 1376x4096 (GPU 0)]
                  (5): Parameter containing: [torch.bfloat16 of size 1376x4096 (GPU 0)]
                  (6): Parameter containing: [torch.bfloat16 of size 1376x4096 (GPU 0)]
                  (7): Parameter containing: [torch.bfloat16 of size 1376x4096 (GPU 0)]
              )
              (weight_down): ParameterList(
                  (0): Parameter containing: [torch.bfloat16 of size 4096x1376 (GPU 0)]
                  (1): Parameter containing: [torch.bfloat16 of size 4096x1376 (GPU 0)]
                  (2): Parameter containing: [torch.bfloat16 of size 4096x1376 (GPU 0)]
                  (3): Parameter containing: [torch.bfloat16 of size 4096x1376 (GPU 0)]
                  (4): Parameter containing: [torch.bfloat16 of size 4096x1376 (GPU 0)]
                  (5): Parameter containing: [torch.bfloat16 of size 4096x1376 (GPU 0)]
                  (6): Parameter containing: [torch.bfloat16 of size 4096x1376 (GPU 0)]
                  (7): Parameter containing: [torch.bfloat16 of size 4096x1376 (GPU 0)]
              )
            )
          )
        )
      )
    )
    (norm): LlamaRMSNorm()
  )
  (lm_head): Linear(in_features=4096, out_features=32000, bias=False)
)
```



 






---- 



---
license: apache-2.0
language:
- en
tags:
- MoE
---


# LLaMA-MoE-v1-3.5B (2/8)

[[💻 Code]](https://github.com/pjlab-sys4nlp/llama-moe) | [[📜 Technical Report]](https://github.com/pjlab-sys4nlp/llama-moe/blob/main/docs/LLaMA_MoE.pdf)

👋 Very nice to meet you here~

❤️ This repo contains the model `LLaMA-MoE-v1-3.5B (2/8)`, which activates 2 out of 8 experts (3.5B parameters).
This model is NOT fine-tuned by instruction pairs, so it may not be good enough to act like a chatbot.

📢 LLaMA-MoE is a series of Mixture-of-Expert (MoE) models based on [LLaMA-2](https://huggingface.co/meta-llama/Llama-2-7b-hf).
You can find the code for training this model at [this repo](https://github.com/pjlab-sys4nlp/llama-moe).

💎 This series of models are obtained by partitioning original LLaMA FFNs into experts and further continual pre-training.
The total model size is only 6.7B parameters, which is very convenient for deployment and research usage.
More details could be found at [our technical report](https://arxiv.org/).

## 🚀 QuickStart

```python
# python>=3.10

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_dir = "llama-moe/LLaMA-MoE-v1-3_5B-2_8"
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16, trust_remote_code=True)
model.eval()
model.to("cuda:0")

input_text = "Suzhou is famous of"
inputs = tokenizer(input_text, return_tensors="pt")
inputs = inputs.to("cuda:0")

pred = model.generate(**inputs, max_length=50, temperature=0.0)
print(tokenizer.decode(pred.cpu()[0], skip_special_tokens=True))
# Suzhou is famous of its beautiful gardens. The most famous one is the Humble Administrator's Garden. It is a classical Chinese garden with a history of more than 600 years. The garden is divided into three
```

## 📊 Performance

| Model                     | \#Activated Experts | \#Experts | \#Activated Params |                                   Links                                   |
| :------------------------ | :-----------------: | :-------: | :----------------: | :-----------------------------------------------------------------------: |
| **LLaMA-MoE-3.0B**        |          2          |    16     |        3.0B        | [[🤗 HF Weights]](https://huggingface.co/llama-moe/LLaMA-MoE-v1-3_0B-2_16) |
| **LLaMA-MoE-3.5B (4/16)** |          4          |    16     |        3.5B        | [[🤗 HF Weights]](https://huggingface.co/llama-moe/LLaMA-MoE-v1-3_5B-4_16) |
| **LLaMA-MoE-3.5B (2/8)**  |          2          |     8     |        3.5B        | [[🤗 HF Weights]](https://huggingface.co/llama-moe/LLaMA-MoE-v1-3_5B-2_8)  |


| Model                                                                                 |   SciQ   |   PIQA   | WinoGrande |  ARC-e   | ARC-c (25) | HellaSwag (10) |  LogiQA  | BoolQ (32) | LAMBADA  | NQ (32)  |  MMLU (5) | Average |
| :------------------------------------------------------------------------------------ | :------: | :------: | :--------: | :------: | :--------: | :------------: | :------: | :--------: | :------: | :------: | :-------: | :-----: |
| [OPT-2.7B](https://huggingface.co/facebook/opt-2.7b)                                  |   78.9   |   74.8   |    60.8    |   54.4   |    34.0    |      61.4      |   25.8   |    63.3    |   63.6   |   10.7   |   25.8    |  50.3   |
| [Pythia-2.8B](https://huggingface.co/EleutherAI/pythia-2.8b)                          |   83.2   |   73.6   |    59.6    |   58.8   |    36.7    |      60.7      |   28.1   |    65.9    |   64.6   |   8.7    |   26.8    |  51.5   |
| [INCITE-BASE-3B](https://huggingface.co/togethercomputer/RedPajama-INCITE-Base-3B-v1) |   85.6   |   73.9   |    63.5    |   61.7   |    40.3    |      64.7      |   27.5   |    65.8    |   65.4   |   15.2   |   27.2    |  53.7   |
| [Open-LLaMA-3B-v2](https://huggingface.co/openlm-research/open_llama_3b_v2)           |   88.0   |   77.9   |    63.1    |   63.3   |    40.1    |      71.4      |   28.1   |    69.2    |   67.4   |   16.0   |   26.8    |  55.6   |
| [Sheared-LLaMA-2.7B](https://huggingface.co/princeton-nlp/Sheared-LLaMA-2.7B)         |   87.5   |   76.9   |    65.0    |   63.3   |    41.6    |      71.0      |   28.3   |    73.6    |   68.3   |   17.6   | **27.3**  |  56.4   |
| **LLaMA-MoE-3.0B**                                                                    |   84.2   |   77.5   |    63.6    |   60.2   |    40.9    |      70.8      | **30.6** |    71.9    |   66.6   |   17.0   |   26.8    |  55.5   |
| **LLaMA-MoE-3.5B (4/16)**                                                             |   87.6   | **77.9** |    65.5    | **65.6** |  **44.2**  |    **73.3**    |   29.7   |  **75.0**  | **69.5** | **20.3** |   26.8    |  57.7   |
| **LLaMA-MoE-3.5B (2/8)**                                                              | **88.4** |   77.6   |  **66.7**  |   65.3   |    43.1    |    **73.3**    |   29.6   |    73.9    |   69.4   |   19.8   |   27.0    |  57.6   |

## 📖 Details

Training Data: 200B tokens from [SlimPajama](https://www.cerebras.net/blog/slimpajama-a-627b-token-cleaned-and-deduplicated-version-of-redpajama) with the same data sampling weights as [Sheared LLaMA](https://arxiv.org/abs/2310.06694).

## 📃 Citation

```bibtex
@article{llama-moe,
  title={LLaMA-MoE: Building Mixture-of-Experts from LLaMA with Continual Pre-training},
  author={LLaMA-MoE Team},
  journal={arXiv},
  year={2023},
  volume={abs/},
  url={https://arxiv.org}
}
```
 

