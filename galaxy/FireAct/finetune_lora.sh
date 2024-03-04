cd finetune/llama_lora
python finetune.py \
--base_model  ../../../../../llama-2-7b/llama-2-7b-chat-hf \
--data_path ../../data/finetune/alpaca_format/hotpotqa.json \
--micro_batch_size 4 \
--num_epochs 1 \
--output_dir ../output_models/lora_models/fireact_llama-2-7b-chat-hf \
--val_set_size 0.01 \
--cutoff_len 512 \
--lora_r 8 \
--lora_alpha 16 \
--lora_dropout 0.05 \
--lora_target_modules '[q_proj,v_proj]' \
--train_on_inputs \
--group_by_length \
--report_to 'tensorboard' \
--logging_dir "tblogs" 