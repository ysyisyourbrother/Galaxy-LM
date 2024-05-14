model=t5base
method=adapters
# for task in   "stsb" "qnli" "sst2" "qqp" "mnli"   "qnli"  "sst2" "qqp" "mnli"
for task in  "sst2"  "stsb" 
do
    bash scripts/${model}/${method}.sh "1" ${task}
done