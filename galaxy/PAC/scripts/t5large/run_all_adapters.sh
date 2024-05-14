model=t5large
method=adapters
# for task in   "stsb" "qnli" "sst2" "qqp" "mnli"
# for task in "cola"  "mrpc" "stsb"    "qnli"  "sst2" "qqp" "mnli" "qnli"  "sst2"   "mnli"
for task in "cola"  "mrpc" "stsb"  "rte"  
do
    bash scripts/${model}/${method}.sh "1" ${task}
done