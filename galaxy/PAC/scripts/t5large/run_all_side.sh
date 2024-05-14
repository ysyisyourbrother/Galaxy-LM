model=t5large
method=side
# for task in   "stsb" "qnli" "sst2" "qqp" "mnli" "qnli"  "sst2"   "mnli"
for task in  "rte"  "cola" "mrpc" "stsb"    
do
    bash scripts/${model}/${method}.sh "0" ${task}
done