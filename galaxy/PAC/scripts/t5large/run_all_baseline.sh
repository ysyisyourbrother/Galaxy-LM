model=t5large
method=baseline
# for task in   "stsb" "qnli" "sst2" "qqp" "mnli"
# for task in "rte" "mrpc" "stsb"  "cola"   "qnli"  "sst2" "qqp" "mnli"  "qnli"  "sst2" "qqp"  
for task in  "rte" "mrpc" "stsb"  "cola"  
do
    bash scripts/${model}/${method}.sh "1" ${task}
done