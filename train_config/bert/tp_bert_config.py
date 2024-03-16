import torch
import os
import json 
from  .bert_config import BertConfig
class TPBertConfig(BertConfig):
    def __init__(self):
        super(TPBertConfig,self).__init__()
        ''' Distributed Configuration '''
        self.tp_num_attention_heads_list =[ int(self.num_attention_heads/2) ,  int(self.num_attention_heads/2)]   # 张量并行环境下当前rank有多少个heads
        self.tp_intermediate_size_list = [int(self.intermediate_size/2)  ,  int(self.intermediate_size/2)]         # TP下MLP两个dense层中间的intermediate state大小
        self.init_method ="tcp://127.0.0.1:23000"                          # torch.dist.init_process_group中使用的master device    
        self.distributed_backend = "gloo"
        self.tp_num_attention_heads = None
        self.tp_intermediate_size = None
        if (sum(self.tp_num_attention_heads_list) != self.num_attention_heads):
            raise ValueError("Sum of tp_num_attention_heads_list must equal to num_attention_heads")
        if (sum(self.tp_intermediate_size_list) != self.intermediate_size):
            raise ValueError("Sum of tp_intermediate_size_list must equal to intermediate_size")

        
    def load_from_json(self,config_file):
        super().load_from_json(config_file)
        if not os.path.exists(config_file):
            raise FileNotFoundError("config file: {} not found".format(config_file))
        with open(config_file, "r") as f:
            config_dict = json.load(f)
            print("==========Updating config from file: ", config_file,"==========")
        # Distributed Configuration
        self.init_method = config_dict["init_method"]                       # torch.dist.init_process_group中使用的master device
        self.distributed_backend = config_dict["distributed_backend"] # 通信后端
        self.tp_num_attention_heads_list = config_dict["tp_num_attention_heads_list"]  # 张量并行当前rank的head数]  # 张量并行当前rank的head数
        self.tp_intermediate_size_list =  config_dict["tp_intermediate_size_list"] # TP下MLP两个dense层中间的intermediate state大小
        if (sum(self.tp_num_attention_heads_list) != self.num_attention_heads):
            raise ValueError("Sum of tp_num_attention_heads_list must equal to num_attention_heads")
        if (sum(self.tp_intermediate_size_list) != self.intermediate_size):
            raise ValueError("Sum of tp_intermediate_size_list must equal to intermediate_size")
    
    def update_tp_config(self, args):
        rank = args.rank
        world = args.world
        if len(self.tp_num_attention_heads_list) != world:
            raise ValueError("len(tp_num_attention_heads_list) must equal to world")
        if len(self.tp_intermediate_size_list) != world:
            raise ValueError("len(tp_intermediate_size_list) must equal to world")
        self.tp_num_attention_heads = self.tp_num_attention_heads_list[rank]
        self.tp_intermediate_size = self.tp_intermediate_size_list[rank]
        

config = TPBertConfig()