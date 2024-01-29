#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence,Tuple
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import transformers
from transformers import Trainer
from transformers import  AutoTokenizer
from llama_moe_predict.modeling_llama_moe_with_predict import LlamaMoEForCausalLMPredict
from torch.utils.tensorboard import SummaryWriter

import sys 
import alpaca.utils as utils
from  alpaca.utils  import get_parameter_number

import time
import datetime
from datetime import timedelta
device = "cuda"
 

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """
    Resize tokenizer and embedding.
    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    调整预训练模型的分词器（tokenizer）和嵌入层（embedding）的大小
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """
    Tokenize a list of strings.
    分词
    
    """
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """
    Preprocess the data by tokenizing.
    
    """
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        list_data_dict = utils.jload(data_path)

        logging.warning("Formatting inputs...")
        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        sources = [
            prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
            for example in list_data_dict
        ]
        targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)

 
writer = SummaryWriter('./logs/%s/' % datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
def init_predict_gate (model):
    #用下一层gate初始化当前层的predict_gate
    print("init predict gate with next layer's gate")
    num_hidden_layers = model.config.num_hidden_layers
    for i in range(num_hidden_layers-1):
        model.model.layers[i].mlp.predict_gate.gate_network.load_state_dict(model.model.layers[i+1].mlp.gate.gate_network.state_dict())

def compute_predict_score (layer, predict, target, num_experts, num_selects):
    _, top_indices_predict = predict.topk(min(num_selects + 1,  num_experts), dim=1)  # 选择并排序前k+1个权重
    top_k_indices_predict = top_indices_predict[:, :num_selects]
    
    _, top_indices_target = target.topk(min(num_selects + 1,  num_experts), dim=1)  # 选择并排序前k+1个权重
    top_k_indices_target = top_indices_target[:, :num_selects]
    # 统计每一行中 predict 中的元素在对应行中的 target 中的个数
    counts = []
    for i in range(top_k_indices_predict.size(0)):
        # 一个token预测对的数量 
        row_count = torch.sum(torch.isin(top_k_indices_predict[i], top_k_indices_target[i])).item()
        counts.append(row_count)
       
    ratio = [i / num_selects for i in counts]
    avg_ratio =  sum(ratio)/ len(ratio)
    # avg_ratio,sum(counts),top_indices_target.numel()
    print("ratio = ",ratio)
    print("avg_ratio_layer {} = {}, num_of_right_predict = {}, total = {}".format(layer,avg_ratio, sum(counts), top_k_indices_target.numel()))
    writer.add_scalar("avg_ratio_layer_{}".format(layer), avg_ratio)
    # writer.add_scalar("layer_{}_right_predict".format(layer), sum(counts))
    # writer.add_scalar("layer_{}_total".format(layer), top_k_indices_target.numel())
    writer.add_scalar("correct_predict_ratio_layer_{}".format(layer), sum(counts) / top_k_indices_target.numel())

class CustomTrainer(Trainer):
    '''
    rewrite compute_loss
    e.g.
        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.pop("labels")
            # forward pass
            outputs = model(**inputs)
            logits = outputs.get("logits")
            # compute custom loss (suppose one has 3 labels with different weights)
            loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0, 3.0], device=model.device))
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            return (loss, outputs) if return_outputs else loss
    '''
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        all_gate_inputs: Optional[Tuple[torch.FloatTensor]] = outputs.get("all_gate_inputs", None)
        all_gate_outputs: Optional[Tuple[torch.FloatTensor]] = outputs.get("all_gate_outputs", None)
        assert all_gate_inputs is not None and all_gate_outputs is not None, "all_gate_inputs and all_gate_outputs must be specified"
        num_layers = len(all_gate_inputs)
        loss = 0.0
        for i in range(num_layers-1):
            x = all_gate_inputs[i].to(torch.bfloat16)
            target = all_gate_outputs[i+1].to(torch.bfloat16)
            y = model.model.layers[i].mlp.predict_gate.gate_network(x)
            criterion =  nn.MSELoss()
            layer_loss = criterion(y, target)
            loss += layer_loss
            print("loss_layer_{} = {}".format(i, loss.item()))
            writer.add_scalar("loss_layer_{}".format(i), layer_loss.item() )
            compute_predict_score(i, y, target, model.config.num_experts, model.config.num_selects)
        writer.add_scalar("total_loss" , loss.item() )
        return (loss, outputs) if return_outputs else loss
  
    
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        '''
        only save trainabel parameters
        currently only predict_gate
        '''
        assert output_dir is not None, "output_dir must be specified"
        full_state_dict = self.model.state_dict()
        # 提取需要保存的模块的权重
        module_to_save_state = {k: v for k, v in full_state_dict.items() if 'predict_gate' in k}
        new_state_dict = { k: v for k, v in module_to_save_state.items()}
        print("save model: ","predict_gate")
        self._save(output_dir, state_dict=new_state_dict)



def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    print("finish load tokenizer")
    start_time = time.time()
    with torch.device( device):
        model = LlamaMoEForCausalLMPredict.from_pretrained(model_args.model_name_or_path, torch_dtype=torch.bfloat16)
    model.to(device)
    elapse_time = timedelta(seconds=int(round( time.time() - start_time )))
    print("finish load model, time cost: ", elapse_time)
    print(get_parameter_number(model))
    
    for name, param in model.named_parameters():
        if "predict_gate" not in name:  #  除了predict_gate
            param.requires_grad = False
    print(get_parameter_number(model))
    
    init_predict_gate(model)
    # load predict_gate
    # model.load_state_dict(torch.load("./predict_output/pytorch_model.bin"))
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )
    # prepare data
    start_time = time.time()
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    elapse_time = timedelta(seconds=int(round( time.time() - start_time )))
    print("finish preparing data, time cost: ", elapse_time)
    # prepare trainer, data_module里面有train_dataset，eval_dataset，data_collator
    trainer = CustomTrainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    # trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    print("begin traning")
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
