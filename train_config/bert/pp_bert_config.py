import torch
import os
import json 
from  .bert_config import BertConfig
class PPBertConfig(BertConfig):
    def __init__(self):
        super(PPBertConfig,self).__init__()
        # Distributed Configuration 
        self.init_method = "tcp://127.0.0.1:23000"                         # torch.dist.init_process_group中使用的master device    
        self.distributed_backend = "gloo"
        self.stage_num_hidden_layers_list = [1,1]
        if sum(self.stage_num_hidden_layers_list) != self.num_hidden_layers:
            raise ValueError("sum of stage_hidden_layers_num_list should be equal to num_hidden_layers")
        self.num_microbatches = 4
        self.num_iterations = 4
        # Pipeline Configuration
        self.stage = None
        self.total_stage = None
        self.pre_rank = None
        self.next_rank = None
        self.is_first_stage = None
        self.is_last_stage = None
        self.num_pp_hidden_layers = None 

    def load_from_json(self,config_file):
        super().load_from_json(config_file)
        if not os.path.exists(config_file):
            raise FileNotFoundError("config file: {} not found".format(config_file))
        with open(config_file, "r") as f:
            config_dict = json.load(f)
        ''' Distributed Configuration '''
        self.init_method = config_dict["init_method"]                       # torch.dist.init_process_group中使用的master device
        self.distributed_backend = config_dict["distributed_backend"] # 通信后端
        self.stage_num_hidden_layers_list = config_dict["stage_num_hidden_layers_list"]
        if sum(self.stage_num_hidden_layers_list) != self.num_hidden_layers:
            raise ValueError("sum of stage_hidden_layers_num_list should be equal to num_hidden_layers")
        self.num_microbatches = config_dict["num_microbatches"]
        self.num_iterations = config_dict["num_iterations"]
 
    def update_pp_stage_config(self, args):
        self.stage = args.rank
        self.total_stage =args.world 
        if self.total_stage==1:
            raise ValueError("total_stage should > 1")
        if self.total_stage != len(self.stage_num_hidden_layers_list):
            raise ValueError("total_stage != len(stage_num_hidden_layers_list)")
        self.num_pp_hidden_layers = self.stage_num_hidden_layers_list[self.stage]
        self.pre_rank = None if self.stage == 0 else self.stage - 1
        self.next_rank = None if self.stage  == self.total_stage -1 else self.stage + 1
        self.is_first_stage = (self.stage ==0 )  # 第一个需要过embedding
        self.is_last_stage =  (self.stage  ==  self.total_stage - 1)    # 最后一个增加过分类头
        print("update PP stage config: stage={}, total_stage={}, num_pp_hidden_layers={}".format(self.stage, self.total_stage, self.num_pp_hidden_layers))
        
config = PPBertConfig()