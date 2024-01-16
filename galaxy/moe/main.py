# python>=3.10

import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from llama_moe.modeling_llama_moe import LlamaMoEForCausalLM


model_dir = "./llama_moe/"
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
with torch.device("cuda:0"):
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16, trust_remote_code=True)
model.eval()
model.to("cuda:0")

input_text = "Suzhou is famous of its beautiful gardens. The most famous one is the Humble Administrator's Garden. It is a classical Chinese garden with a history of more than 600 years. The garden is divided into three"
inputs = tokenizer(input_text, return_tensors="pt")
inputs = inputs.to("cuda:0")
print("inputs shape:", inputs["input_ids"].shape)

pred = model.generate(**inputs, max_length=51, temperature=0.0)
print(tokenizer.decode(pred.cpu()[0], skip_special_tokens=True))
# Suzhou is famous of its beautiful gardens. The most famous one is the Humble Administrator's Garden. It is a classical Chinese garden with a history of more than 600 years. The garden is divided into three
