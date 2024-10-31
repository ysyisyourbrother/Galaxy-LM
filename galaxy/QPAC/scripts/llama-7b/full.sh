model_name=llama-7b
method=full
folder_name=all_output_logs/${model_name}
if [ ! -d ${folder_name} ] ; then
    mkdir -p ${folder_name}
fi
source scripts/env.sh

config_file=configs/${model_name}/${method}.json
for seed in 42
do
    rm -r outputs/${model_name}/${method}/
    python scripts/update_scripts_for_given_input.py ${config_file} model_name_or_path str  /root/nfs/codespace/llm-models/Llama-2-7b-hf 
    python scripts/update_scripts_for_given_input.py ${config_file} task_name str $2

    python scripts/update_scripts_for_given_input.py ${config_file} seed int $seed
    python scripts/update_scripts_for_given_input.py ${config_file} num_train_epochs int ${num_epochs[$2]}
    python scripts/update_scripts_for_given_input.py ${config_file} output_dir str outputs/${model_name}/${method}

    python scripts/update_scripts_for_given_input.py ${config_file} full_finetune str True
    python scripts/update_scripts_for_given_input.py ${config_file} lst str False
    python scripts/update_scripts_for_given_input.py ${config_file} qst str False

    CUDA_VISIBLE_DEVICES=$1 python run_glue.py  $config_file

    cp outputs/${model_name}/${method}/all_results.json  all_output_logs/${model_name}/${method}_epoch_${num_epochs[$2]}_$2@${seed}.json

done