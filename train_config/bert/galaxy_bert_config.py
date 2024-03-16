import torch
import os
import json
from  .bert_config import BertConfig
class GalaxyBertConfig(BertConfig):
    def __init__(self):
        super(GalaxyBertConfig,self).__init__()
        ''' Distributed Configuration '''
        # CON: SP / None
        self.con_parallel_method = "None"  # 
        self.seq_scatter_list = [20,12] 
        # ATT:TP 
        self.att_parallel_method = "TP"
        self.tp_num_attention_heads_list =[ int(self.num_attention_heads/2) ,  int(self.num_attention_heads/2)]       # 张量并行环境下当前rank有多少个heads
        self.tp_num_attention_heads = None
        # MLP:TP
        self.mlp_parallel_method = "SP"
        self.tp_intermediate_size_list =[int(self.intermediate_size/2)  ,  int(self.intermediate_size/2)]               # TP下MLP两个dense层中间的intermediate state大小
        self.tp_intermediate_size = None
        if (sum(self.tp_num_attention_heads_list) != self.num_attention_heads):
            raise ValueError("Sum of tp_num_attention_heads_list must equal to num_attention_heads")
        if (sum(self.tp_intermediate_size_list) != self.intermediate_size):
            raise ValueError("Sum of tp_intermediate_size_list must equal to intermediate_size")
        # init process
        self.init_method = "tcp://127.0.0.1:23000"                         # torch.dist.init_process_group中使用的master device    
        self.distributed_backend = "gloo"


    def load_from_json(self,config_file):
        super().load_from_json(config_file)
        if not os.path.exists(config_file):
            raise FileNotFoundError("config file: {} not found".format(config_file))
        with open(config_file, "r") as f:
            config_dict = json.load(f)

        # Distributed Configuration
        # CON: SP
        self.con_parallel_method = config_dict["con_parallel_method"]
        self.seq_scatter_list = config_dict["seq_scatter_list"]
        # ATT:TP
        self.att_parallel_method = config_dict["att_parallel_method"]
        self.tp_num_attention_heads_list = config_dict["tp_num_attention_heads_list"]
        # MLP:TP
        self.mlp_parallel_method = config_dict["mlp_parallel_method"]
        self.tp_intermediate_size_list = config_dict["tp_intermediate_size_list"]
        if (sum(self.tp_num_attention_heads_list) != self.num_attention_heads):
            raise ValueError("Sum of tp_num_attention_heads_list must equal to num_attention_heads")
        if (sum(self.tp_intermediate_size_list) != self.intermediate_size):
            raise ValueError("Sum of tp_intermediate_size_list must equal to intermediate_size")
        # init process
        self.init_method = config_dict["init_method"]
        self.distributed_backend = config_dict["distributed_backend"]
    def update_galaxy_config(self, args):
        rank = args.rank
        world = args.world
        self.tp_num_attention_heads = self.tp_num_attention_heads_list[rank]
        self.tp_intermediate_size = self.tp_intermediate_size_list[rank]
        
config = GalaxyBertConfig()