model=t5large
method=lora
# for task in   "stsb" "qnli" "sst2" "qqp" "mnli"
for task in       "rte" "cola" "stsb" "mrpc"
do
    bash scripts/${model}/${method}.sh "1" ${task}
done