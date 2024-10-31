model_name=opt-1.3b
method=lst
folder_name=all_output_logs/${model_name}
if [ ! -d ${folder_name} ] ; then
    mkdir -p ${folder_name}
fi
source scripts/env.sh

config_file=configs/${model_name}/${method}.json
for seed in 42
do
    rm -r outputs/${model_name}/${method}/
    python scripts/update_scripts_for_given_input.py ${config_file} model_name_or_path str /root/nfs/codespace/llm-models/opt/opt-1.3b
    python scripts/update_scripts_for_given_input.py ${config_file} task_name str $2

    python scripts/update_scripts_for_given_input.py ${config_file} seed int $seed
    python scripts/update_scripts_for_given_input.py ${config_file} num_train_epochs int ${num_epochs[$2]}
    python scripts/update_scripts_for_given_input.py ${config_file} output_dir str outputs/${model_name}/${method}

    python scripts/update_scripts_for_given_input.py ${config_file} full_finetune str false
    python scripts/update_scripts_for_given_input.py ${config_file} lst str true
    python scripts/update_scripts_for_given_input.py ${config_file} qst str false
    python scripts/update_scripts_for_given_input.py ${config_file} lora_r int 64
    python scripts/update_scripts_for_given_input.py ${config_file} lora_alpha int 16
    python scripts/update_scripts_for_given_input.py ${config_file} double_quant str true
    python scripts/update_scripts_for_given_input.py ${config_file} quant_type str nf4
    python scripts/update_scripts_for_given_input.py ${config_file} bf16 str false
    python scripts/update_scripts_for_given_input.py ${config_file} fp16 str false
    python scripts/update_scripts_for_given_input.py ${config_file} bits int 4



    CUDA_VISIBLE_DEVICES=$1 python run_glue.py  $config_file

    cp outputs/${model_name}/${method}/all_results.json  all_output_logs/${model_name}/${method}_epoch_${num_epochs[$2]}_$2@${seed}_fp32.json

done