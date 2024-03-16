import torch
import os
import json 
from  .bert_config import BertConfig

class SPBertConfig(BertConfig):
    def __init__(self):
        super(SPBertConfig, self).__init__()
        ''' Distributed Configuration '''
        self.init_method = "tcp://127.0.0.1:23000"                         # torch.dist.init_process_group中使用的master device    
        self.distributed_backend = "gloo"
        self.seq_scatter_list = [20,12]                              # 包含sequence被划分到每个设备上的长度
        assert sum(self.seq_scatter_list) == self.pad_size      # 要求总和等于seq_len

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
        self.seq_scatter_list = config_dict["seq_scatter_list"]
        assert sum(self.seq_scatter_list) == self.pad_size      # 要求总和等于seq_len
config = SPBertConfig()