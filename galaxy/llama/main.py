from transformers import  LlamaTokenizer
from src.modeling_llama import LlamaForCausalLM
import torch

model_dir = "../../../llama-7b-hf/llama_7b_hf_weight"
tokenizer = LlamaTokenizer.from_pretrained(model_dir)
with torch.device("cuda:0"):
    model = LlamaForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16)
model.eval()
model.to("cuda:0")
print(model)
input_text = "Suzhou is famous of its beautiful gardens. The most famous one is the Humble Administrator's Garden. It is a classical Chinese garden with a history of more than 600 years. The garden is divided into three"
inputs = tokenizer(input_text, return_tensors="pt")
inputs = inputs.to("cuda:0")
print("inputs shape:", inputs["input_ids"].shape)

pred = model.generate(**inputs, max_length=51, temperature=0.0)
print(tokenizer.decode(pred.cpu()[0], skip_special_tokens=True))
# Suzhou is famous of its beautiful gardens. The most famous one is the Humble Administrator's Garden. It is a classical Chinese garden with a history of more than 600 years. The garden is divided into three