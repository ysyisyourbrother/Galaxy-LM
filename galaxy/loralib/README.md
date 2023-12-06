# LoRA: Low-Rank Adaptation of Large Language Models


## Quickstart

 1. Installing `loralib` is simply
 ```bash
 pip install loralib
 # Alternatively
 # pip install git+https://github.com/microsoft/LoRA
 ```

 2. You can choose to adapt some layers by replacing them with counterparts implemented in `loralib`. We only support `nn.Linear`, `nn.Embedding`, and `nn.Conv2d` for now. We also support a `MergedLinear` for cases where a single `nn.Linear` represents more than one layers, such as in some implementations of the attention `qkv` projection (see Additional Notes for more).
 ```python
 # ===== Before =====
 # layer = nn.Linear(in_features, out_features)

 # ===== After ======
 import loralib as lora
 # Add a pair of low-rank adaptation matrices with rank r=16
 layer = lora.Linear(in_features, out_features, r=16)
 ```

 3. Before the training loop begins, mark only LoRA parameters as trainable.
 ```python
 import loralib as lora
 model = BigModel()
 # This sets requires_grad to False for all parameters without the string "lora_" in their names
 lora.mark_only_lora_as_trainable(model)
 # Training loop
 for batch in dataloader:
    ...
 ```
 4. When saving a checkpoint, generate a `state_dict` that only contains LoRA parameters.
 ```python
 # ===== Before =====
 # torch.save(model.state_dict(), checkpoint_path)
 # ===== After =====
 torch.save(lora.lora_state_dict(model), checkpoint_path)
 ```
 5. When loading a checkpoint using `load_state_dict`, be sure to set `strict=False`.
 ```python
 # Load the pretrained checkpoint first
 model.load_state_dict(torch.load('ckpt_pretrained.pt'), strict=False)
 # Then load the LoRA checkpoint
 model.load_state_dict(torch.load('ckpt_lora.pt'), strict=False)
 ```

#### Now training can proceed as usual.

## Additional Notes

1. While we focus on a simple yet effect setup, namely adapting only the `q` and `v` projection in a Transformer, in our examples, LoRA can be apply to any subsets of pre-trained weights. We encourage you to explore different configurations, such as adapting the embedding layer by replacing `nn.Embedding` with `lora.Embedding` and/or adapting the MLP layers. It's very likely that the optimal configuration varies for different model architectures and tasks.

2. Some Transformer implementation uses a single `nn.Linear` for the projection matrices for query, key, and value. If one wishes to constrain the rank of the updates to the individual matrices, one has to either break it up into three separate matrices or use `lora.MergedLinear`. Make sure to modify the checkpoint accordingly if you choose to break up the layer.
```python
# ===== Before =====
# qkv_proj = nn.Linear(d_model, 3*d_model)
# ===== After =====
# Break it up (remember to modify the pretrained checkpoint accordingly)
q_proj = lora.Linear(d_model, d_model, r=8)
k_proj = nn.Linear(d_model, d_model)
v_proj = lora.Linear(d_model, d_model, r=8)
# Alternatively, use lora.MergedLinear (recommended)
qkv_proj = lora.MergedLinear(d_model, 3*d_model, r=8, enable_lora=[True, False, True])
```
3. Training bias vectors in tandem with LoRA might be a cost-efficient way to squeeze out extra task performance (if you tune the learning rate carefully). While we did not study its effect thoroughly in our paper, we make it easy to try in `lora`. You can mark some biases as trainable by passing "all" or "lora_only" to `bias=` when calling `mark_only_lora_as_trainable`. Remember to pass the corresponding `bias=` argument to `lora_state_dict` when saving a checkpoint.
```python
# ===== Before =====
# lora.mark_only_lora_as_trainable(model) # Not training any bias vectors
# ===== After =====
# Training all bias vectors associated with modules we apply LoRA to 
lora.mark_only_lora_as_trainable(model, bias='lora_only')
# Alternatively, we can train *all* bias vectors in the model, including LayerNorm biases
lora.mark_only_lora_as_trainable(model, bias='all')
# When saving a checkpoint, use the same bias= ('all' or 'lora_only')
torch.save(lora.lora_state_dict(model, bias='all'), checkpoint_path)
```
4. Calling `model.eval()` will trigger the merging of LoRA parameters with the corresponding pretrained ones, which eliminates additional latency for subsequent forward passes. Calling `model.train()` again will undo the merge. This can be disabled by passing `merge_weights=False` to LoRA layers.

## Contact
Please contact us or post an issue if you have any questions.

For questions related to the package `loralib`:
* Edward Hu (edward@edwardjhu.com)
* Phillip Wallis (phwallis@microsoft.com)
* Weizhu Chen (wzchen@microsoft.com)

The GPT-2 example:
* Phillip Wallis (phwallis@microsoft.com)
* Yelong Shen (yeshe@microsoft.com)

The RoBERTa/DeBERTa example:
* Lu Wang (luw@microsoft.com)
 
