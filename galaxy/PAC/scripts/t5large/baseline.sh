# This scripts trains full finetuning method.
# For smaller datasets of GLUE (mrpc, cola, and stsb), we set the `num_train_epochs` to 20,
# for other larger datasets in GLUE we used `num_train_epochs` of 3. 
model_name=t5-large
method=baseline
folder_name=all_output_logs/${model_name}
if [ ! -d ${folder_name} ] ; then
    mkdir -p ${folder_name}
fi

source scripts/env.sh

file_name=${model_name}/${method}

for seed in 0 
do
    rm -r outputs/${model_name}/${method}/
    python scripts/update_scripts_for_given_input.py configs/${file_name}.json model_name_or_path str ../../../llm-models/t5/google-t5-large
    python scripts/update_scripts_for_given_input.py configs/${file_name}.json tokenizer_name str ../../../llm-models/t5/google-t5-large
    python scripts/update_scripts_for_given_input.py configs/${file_name}.json task_name str $2
    python scripts/update_scripts_for_given_input.py configs/${file_name}.json eval_dataset_name str $2
    python scripts/update_scripts_for_given_input.py configs/${file_name}.json test_dataset_name str $2

    python scripts/update_scripts_for_given_input.py configs/${file_name}.json seed int $seed
    python scripts/update_scripts_for_given_input.py configs/${file_name}.json num_train_epochs int ${num_epochs[$2]}
    python scripts/update_scripts_for_given_input.py configs/${file_name}.json output_dir str outputs/${model_name}/${method}

    CUDA_VISIBLE_DEVICES=$1 python run_seq2seq.py  configs/${file_name}.json

    cp outputs/${model_name}/${method}/all_results.json  all_output_logs/${model_name}/${method}_epoch_${num_epochs[$2]}_$2@${seed}.json

done