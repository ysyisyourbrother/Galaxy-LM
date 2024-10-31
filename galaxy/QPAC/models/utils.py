
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
import os
from os.path import exists, join, isdir
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence
import numpy as np
from tqdm import tqdm
import logging
import bitsandbytes as bnb
from datasets import load_from_disk
import torch
import transformers
from datasets import load_dataset, Dataset
import evaluate
import importlib
from packaging import version
from packaging.version import parse
import warnings
def is_ipex_available():
    def get_major_and_minor_from_version(full_version):
        
        return str(version.parse(full_version).major) + "." + str(version.parse(full_version).minor)

    _torch_version = importlib.metadata.version("torch")
    if importlib.util.find_spec("intel_extension_for_pytorch") is None:
        return False
    _ipex_version = "N/A"
    try:
        _ipex_version = importlib.metadata.version("intel_extension_for_pytorch")
    except importlib.metadata.PackageNotFoundError:
        return False
    torch_major_and_minor = get_major_and_minor_from_version(_torch_version)
    ipex_major_and_minor = get_major_and_minor_from_version(_ipex_version)
    if torch_major_and_minor != ipex_major_and_minor:
        warnings.warn(
            f"Intel Extension for PyTorch {ipex_major_and_minor} needs to work with PyTorch {ipex_major_and_minor}.*,"
            f" but PyTorch {_torch_version} is found. Please switch to the matching version and run again."
        )
        return False
    return True

def find_all_linear_names(args, model):
    cls = bnb.nn.Linear4bit if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

def print_trainable_parameters(args, model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for name, param in model.named_parameters():
        if "blackbone" in name:
            param.requires_grad = False
        if "model.layer" in name:
            param.requires_grad = False
        all_param += param.numel()
        # if "lm_head" in name:
        #     param.requires_grad = True
        if param.requires_grad:
            # if "qst" not in name and "down" not in name and "z" not in name:
            # print(name)
            trainable_params += param.numel()
    if args.bits == 4: trainable_params /= 2
    print(
        f"trainable params: {trainable_params} || "
        f"all params: {all_param} || "
        f"trainable: {100 * trainable_params / all_param}"
    )
     


def smart_tokenizer_and_embedding_resize(
        special_tokens_dict: Dict,
        tokenizer: transformers.PreTrainedTokenizer,
        model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        if  model.get_input_embeddings()==None:
            raise ValueError("model.get_input_embeddings()==None")
        if model.get_output_embeddings()==None:
            raise ValueError("model.get_output_embeddings()==None")
        input_embeddings_data = model.get_input_embeddings().weight.data
        output_embeddings_data = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings_data[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings_data[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings_data[-num_new_tokens:] = input_embeddings_avg
        output_embeddings_data[-num_new_tokens:] = output_embeddings_avg





def get_last_checkpoint(checkpoint_dir):
    if isdir(checkpoint_dir):
        is_completed = exists(join(checkpoint_dir, 'completed'))
        if is_completed: return None, True  # already finished
        max_step = 0
        for filename in os.listdir(checkpoint_dir):
            if isdir(join(checkpoint_dir, filename)) and filename.startswith('checkpoint'):
                max_step = max(max_step, int(filename.replace('checkpoint-', '')))
        if max_step == 0: return None, is_completed  # training started, but no checkpoint
        checkpoint_dir = join(checkpoint_dir, f'checkpoint-{max_step}')
        print(f"Found a previous checkpoint at: {checkpoint_dir}")
        return checkpoint_dir, is_completed  # checkpoint found!
    return None, False  # first training


class SavePeftModelCallback(transformers.TrainerCallback):
    def save_model(self, args, state, kwargs):
        print('Saving QST checkpoint...')
        if state.best_model_checkpoint is not None:
            checkpoint_folder = os.path.join(state.best_model_checkpoint, "QST_model")
        else:
            checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")
        
        if not os.path.exists(checkpoint_folder):
            try:
                os.makedirs(checkpoint_folder, exist_ok=True)
            except Exception as e:
                print(f"创建目录失败: {e}")
                raise

        kwargs["model"].save_qst_state(checkpoint_folder)

    def on_save(self, args, state, control, **kwargs):
        self.save_model(args, state, kwargs)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        def touch(fname, times=None):
            with open(fname, 'a'):
                os.utime(fname, times)

        touch(join(args.output_dir, 'completed'))
        self.save_model(args, state, kwargs)
        
        
def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'total_para': total_num, 'trainable_para': trainable_num} 

def mark_only_side_as_trainable(model):
    # currently only used for QSTT5
    for name, param in model.named_parameters():
        if any(substring in name for substring in ["side", "qst", "z", "downsample", "upsample"]):
            param.requires_grad = True
        elif any(substring in name for substring in ["classification_head","lm_head"]):  
            param.requires_grad = True
        if any(substring in name for substring in ["encoder.block", "decoder.block"]):
            param.requires_grad = False
        if any(substring in name for substring in ["encoder.layers", "decoder.layers"]):
            param.requires_grad = False    
    return  model

def get_model_memory_usage(model):
    # 定义不同数据类型占用的字节数
    dtype_to_bytes = {
        torch.float32: 4,
        torch.float16: 2,
        torch.bfloat16: 2,
        torch.int8: 1,
        torch.int16: 2,
        torch.int32: 4,
        torch.int64: 8,
        torch.uint8: 1,
        
    }

    # 总内存大小
    total_memory_bytes = 0
    # 遍历模型参数
    for name, param in model.named_parameters():
        # 获取每个参数的元素数量
        num_elements = param.numel()
        # 获取参数的数据类型
        param_dtype = param.dtype
        # 根据数据类型确定每个参数元素的字节数
        if param_dtype not in dtype_to_bytes:
            raise ValueError(f"Unsupported data type: {param_dtype}")
        bytes_per_param = dtype_to_bytes[param_dtype]

        # 计算该参数的内存占用
        total_memory_bytes += num_elements * bytes_per_param

    # 将字节转换为GB
    total_memory_gb = total_memory_bytes / (1024 * 1024*1024)

    return total_memory_gb
 