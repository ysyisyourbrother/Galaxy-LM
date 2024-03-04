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