model=t5large
# for task in   "stsb" "qnli" "sst2" "qqp" "mnli"
# for task in "rte" "mrpc" "stsb"  "cola"   "qnli"  "sst2" "qqp" "mnli"  "qnli"  "sst2" "qqp"  
for task in  "rte" "mrpc" "stsb"  "cola"  
do
    for method in "adapters" "side" "lora"
    do
        bash scripts/${model}/${method}.sh "1" ${task}
    done 
done