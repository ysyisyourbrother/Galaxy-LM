import torch
import os
import json 
from  .bert_config import BertConfig
class SideBertConfig(BertConfig):
    def __init__(self):
        super(SideBertConfig,self).__init__()
        #side config
        self.side_reduction_factor = 16
        
    def load_from_json(self,config_file):
        super().load_from_json(config_file)
        if not os.path.exists(config_file):
            raise FileNotFoundError("config file: {} not found".format(config_file))
        with open(config_file, "r") as f:
            config_dict = json.load(f)
            print("========== Updating config from file: ", config_file,"==========")
                # Data Configuration
            self.side_reduction_factor =  config_dict["side_reduction_factor"]
config = SideBertConfig()