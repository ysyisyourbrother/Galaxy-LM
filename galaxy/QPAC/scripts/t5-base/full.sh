model_name=t5-base
method=full
folder_name=all_output_logs/${model_name}
if [ ! -d ${folder_name} ] ; then
    mkdir -p ${folder_name}
fi
source scripts/env.sh

config_file=configs/${model_name}/${method}.json
output_folder=outputs/${model_name}/${method}/

for lr in 2e-5 3e-3 3e-4 1e-4
do
    for seed in 42 1 2
    do
        for bs in 16 32 64 128
        do 
            rm -r outputs/${model_name}/${method}/
            python scripts/update_scripts_for_given_input.py ${config_file} model_name_or_path str /root/nfs/codespace/llm-models/t5/t5-base
            python scripts/update_scripts_for_given_input.py ${config_file} task_name str $2

            python scripts/update_scripts_for_given_input.py ${config_file} seed int $seed
            python scripts/update_scripts_for_given_input.py ${config_file} num_train_epochs int ${num_epochs[$2]}
            python scripts/update_scripts_for_given_input.py ${config_file} output_dir str ${output_folder}

            python scripts/update_scripts_for_given_input.py ${config_file} full_finetune str true
            python scripts/update_scripts_for_given_input.py ${config_file} lst str False
            python scripts/update_scripts_for_given_input.py ${config_file} qst str False

            python scripts/update_scripts_for_given_input.py ${config_file} learning_rate float ${lr}
            python scripts/update_scripts_for_given_input.py ${config_file} per_device_train_batch_size int ${bs}
            python scripts/update_scripts_for_given_input.py ${config_file} per_device_eval_batch_size int 100

            CUDA_VISIBLE_DEVICES=$1 python run_glue.py  $config_file

            cp  ${output_folder}/all_results.json  all_output_logs/${model_name}/${method}_epoch_${num_epochs[$2]}_$2@lr_${lr}_seed_${seed}_bs_${bs}.json
        done
    done
done