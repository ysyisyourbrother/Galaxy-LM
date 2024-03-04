export CUDA_VISIBLE_DEVICES=1
export SERPAPI_API_KEY=aa9b9b5a2cc8decd77d8799827ceda379c0e0f32f5d4c3d5f68e8b79d402eaf4

TASK="hotpotqa" 
MODELPATH="../../../llama-2-7b/fireact_llama_2_7b"
# MODELPATH="finetune/output_models/full_models/fireact_llama-2-7b-chat-hf/checkpoint-6000"
# MODELPATH="../llama-2-7b/llama-2-7b-chat-hf"
if [ "$TASK" = "hotpotqa" ]; then 
    TASK_SPLIT="dev" 
    TASK_END_INDEX=5
elif [ "$TASK" = "strategyqa" ]; then
    TASK_SPLIT="dev" 
    TASK_END_INDEX=200
elif [ "$TASK" = "bamboogle" ]; then
    TASK_SPLIT="test"
    TASK_END_INDEX=100
elif [ "$TASK" = "mmlu" ]; then
    TASK_SPLIT="test"
    TASK_END_INDEX=100
else 
    echo "TASK should be hotpotqa, strategyqa, bamboogle, or mmlu"
    exit 1
fi



python generation.py \
--task $TASK \
--backend llama \
--evaluate \
--random \
--task_split $TASK_SPLIT \
--task_end_index $TASK_END_INDEX \
--modelpath $MODELPATH \
--alpaca_format