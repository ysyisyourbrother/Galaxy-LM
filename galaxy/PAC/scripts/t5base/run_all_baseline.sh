model=t5base
method=baseline
# for task in   "stsb" "qnli" "sst2" "qqp" "mnli"
# for task in "rte" "mrpc" "stsb"  "cola"   "qnli"  "sst2" "qqp" "mnli"
for task in "rte" "cola"  "mrpc" "stsb"    
do
    bash scripts/${model}/${method}.sh "1" ${task}
done