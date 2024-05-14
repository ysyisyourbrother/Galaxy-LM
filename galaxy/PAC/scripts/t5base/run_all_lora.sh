model=t5base
method=lora
# for task in   "stsb" "qnli" "sst2" "qqp" "mnli"   "qnli"  "sst2" "qqp" "mnli"
for task in  "rte" "cola"  
do
    bash scripts/${model}/${method}.sh "1" ${task}
done