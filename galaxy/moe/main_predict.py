# python>=3.10

import torch
from transformers import  AutoTokenizer
from  llama_moe_predict.modeling_llama_moe_with_predict import LlamaMoEForCausalLMPredict

device = "cuda:0"
model_dir = "./llama_moe/"
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
with torch.device(device):
    model = LlamaMoEForCausalLMPredict.from_pretrained(model_dir, torch_dtype=torch.bfloat16, trust_remote_code=True)
model.eval()
model.to(device)
print(model)
input_text = "Suzhou is famous of its beautiful gardens. The most famous one is the Humble Administrator's Garden. It is a classical Chinese garden with a history of more than 600 years. The garden is divided into three"
inputs = tokenizer(input_text, return_tensors="pt")
inputs = inputs.to(device)
print("inputs shape:", inputs["input_ids"].shape)

pred = model.generate(**inputs, max_length=51, temperature=0.0)
print(tokenizer.decode(pred.cpu()[0], skip_special_tokens=True))
# # Suzhou is famous of its beautiful gardens. The most famous one is the Humble Administrator's Garden. It is a classical Chinese garden with a history of more than 600 years. The garden is divided into three

outputs = model(**inputs)
print(outputs.logits.shape)
print(len(outputs.all_gate_inputs))
print(outputs.all_gate_inputs[0].shape)
print(len(outputs.all_gate_outputs))
print(outputs.all_gate_outputs[0].shape)