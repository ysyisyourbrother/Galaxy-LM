# LLaMA
llama: https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/llama.md

src: https://github.com/huggingface/transformers/tree/main/src/transformers/models/llama

llama-7b-hf 权重: Huggingface上 https://huggingface.co/luodian/llama-7b-hf/tree/main 下载 (This works perfectly for transformers>=4.28.0.)

``` lua
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

## Alpaca 数据
从 https://github.com/tatsu-lab/stanford_alpaca
下载
+ alpaca_data.json
+ prompt.txt
note:处理数据时间比较久，测试可以从alpaca_data.json复制几条建一个小的json数据
``` bash
python finetune_alpaca.py \
--model_name_or_path ../../../llama-7b-hf/llama_7b_hf_weight \
--data_path ./alpaca/mini_data.json \
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