# This scripts trains Adapters method.
# For smaller datasets of GLUE (mrpc, cola, and stsb), we set the `num_train_epochs` to 20,
# for other larger datasets in GLUE we used `num_train_epochs` of 3. For all datasets we tried
# with the adapter's bottleneck size of `task_reduction_factor`=[32, 16, 8], and report the 
# results on the test set for the model performing the best on the validation set.
model_name=t5-base
method=lora
folder_name=all_output_logs/${model_name}
if [ ! -d ${folder_name} ] ; then
    mkdir -p ${folder_name}
fi
source scripts/env.sh

file_name=${model_name}/${method}

lora_dim=32

for seed in 0  
do
    rm -r outputs/${model_name}/${method}/
    python scripts/update_scripts_for_given_input.py configs/${file_name}.json model_name_or_path str ../../../llm-models/t5/t5-base
    python scripts/update_scripts_for_given_input.py configs/${file_name}.json tokenizer_name str ../../../llm-models/t5/t5-base
    python scripts/update_scripts_for_given_input.py configs/${file_name}.json task_name str $2
    python scripts/update_scripts_for_given_input.py configs/${file_name}.json eval_dataset_name str $2
    python scripts/update_scripts_for_given_input.py configs/${file_name}.json test_dataset_name str $2

    python scripts/update_scripts_for_given_input.py configs/${file_name}.json seed int $seed
    python scripts/update_scripts_for_given_input.py configs/${file_name}.json num_train_epochs int ${num_epochs[$2]}
    python scripts/update_scripts_for_given_input.py configs/${file_name}.json task_adapter_layers_encoder eval None
    python scripts/update_scripts_for_given_input.py configs/${file_name}.json trainable_encoder_layers eval None

    python scripts/update_scripts_for_given_input.py configs/${file_name}.json task_adapter_layers_decoder eval None
    python scripts/update_scripts_for_given_input.py configs/${file_name}.json trainable_decoder_layers eval None

    python scripts/update_scripts_for_given_input.py configs/${file_name}.json lora_dim int ${lora_dim}

    python scripts/update_scripts_for_given_input.py configs/${file_name}.json output_dir str outputs/${model_name}/${method}

    CUDA_VISIBLE_DEVICES=$1 python run_seq2seq.py  configs/${file_name}.json 

    cp outputs/${model_name}/${method}/all_results.json  all_output_logs/${model_name}/${method}_r_${lora_dim}_epoch_${num_epochs[$2]}_$2@${seed}.json

done
