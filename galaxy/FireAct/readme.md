# FireAct

- [FirAct](https://github.com/anchen1011/FireAct)
- Released model: [forestai/fireact_llama_2_7b](https://huggingface.co/forestai/fireact_llama_2_7b), [forestai/fireact_llama_2_7b_lora](https://huggingface.co/forestai/fireact_llama_2_7b_lora)

## Data

Fine-tuning 数据集: `data/finetune`。数据 gpt4 生成的任务解决轨迹，然后转换成 alpaca_format 的格式。 multimethod 指的是使用不同的 prompts （CoTton、ReAct、Reflexion）。

- hotpotqa.json
- hotpotqa_multimethod.json
- multitask.json
- multitask_multimethod.json

测试数据集: `data/`

- [HotpotQA](https://hotpotqa.github.io) data in `hotpotqa/`.

- [Bamboogle](https://docs.google.com/spreadsheets/d/1jwcsA5kE4TObr9YHn9Gc-wQHYjTbLhDGx6tmIzMhl_U/edit#gid=0) data in `bamboogle/`.

- [StrategyQA](https://allenai.org/data/strategyqa) data in `strategyqa/`.

- [MMLU](https://github.com/hendrycks/test) multiple choice subset in `mmlu/`.

## Supervised Fine-tuning

full model 代码: [tatsu-lab/stanford_alpaca](https://github.com/tatsu-lab/stanford_alpaca)

lora 代码: [tloen/alpaca-lora](https://github.com/tloen/alpaca-lora)

### Full Model

`finetnue_full.sh`

```shell
CUDA_VISIBLE_DEVICES=1
cd finetune/llama_full
python finetune.py \
--model_name_or_path  "../../../../../llama-2-7b/llama-2-7b-chat-hf" \
--data_path ../../data/finetune/alpaca_format/multitask_multimethod.json \
--bf16 True \
--output_dir ../output_models/full_models/fireact_llama-2-7b-chat-hf \
--num_train_epochs 1  \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 1 \
--gradient_accumulation_steps 8 \
--evaluation_strategy "no" \
--save_strategy "steps" \
--save_steps 2000 \
--save_total_limit 1 \
--learning_rate 2e-3 \
--weight_decay 0. \
--warmup_ratio 0.03 \
--lr_scheduler_type "cosine" \
--logging_steps 1 \
--report_to "tensorboard" \
--tf32 True \
--logging_dir "tblogs"
```

## Evaluation / Genration

这个过程需要[`SERPAPI_API_KEY`](https://serpapi.com/),调用 api 过程存在`tool/search.json`

最后保存 trajectory 在`traj`,和输出 em (exact match，评价模型得到的答案，与真实答案完全匹配的比率)

### script

```shell
export CUDA_VISIBLE_DEVICES=1
export SERPAPI_API_KEY="your_api_key"

TASK="bamboogle"
MODELPATH="../llama-2-7b/fireact_llama_2_7b"
# MODELPATH="../llama-2-7b/llama-2-7b-chat-hf"
# MODELPATH="finetune/output_models/full_models/fireact_llama-2-7b-chat-hf/checkpoint-6000"

if [ "$TASK" = "hotpotqa" ]; then
    TASK_SPLIT="dev"
    TASK_END_INDEX=500
elif [ "$TASK" = "strategyqa" ]; then
    TASK_SPLIT="dev"
    TASK_END_INDEX=200
elif [ "$TASK" = "bamboogle" ]; then
    TASK_SPLIT="test"
    TASK_END_INDEX=100
elif [ "$TASK" = "mmlu" ]; then
    TASK_SPLIT="test"
    TASK_END_INDEX=100
else
    echo "TASK should be hotpotqa, strategyqa, bamboogle, or mmlu"
    exit 1
fi
```
