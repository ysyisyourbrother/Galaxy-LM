import torch
import os
import json 

class BertConfig():
    def __init__(self):
        ''' Data Configuration '''
        # 长截短补
        self.pad_size = 32
        # 训练、验证、测试集数据路径
        self.train_path = "dataset/THUCNews/data/train.txt"
        self.dev_path = "dataset/THUCNews/data/dev.txt"
        self.test_path = "dataset/THUCNews/data/test.txt"
        self.vocab_path = "dataset/THUCNews/vocab.txt"

        ''' Training Configuration '''
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备
        # self.device = "cpu"
        self.num_epochs = 3                                             # epoch数
        self.batch_size = 10                                           # mini-batch大小
        self.pad_size = 32                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 5e-5       
        self.class_list = [x.strip() for x in open(
            "dataset/THUCNews/data/class.txt").readlines()]                                # 类别名单
        self.num_classes = len(self.class_list)                         # 类别数

        ''' Bert Configuration '''
        # 模型参数
        self.attention_probs_dropout_prob = 0.1
        self.directionality = "bidi"
        self.hidden_act = "gelu"
        self.hidden_dropout_prob = 0.1
        self.initializer_range = 0.02
        self.layer_norm_eps = 1e-12
        self.max_position_embeddings = 512
        self.model_type = "bert"
        self.pad_token_id = 0

        # 修改模型大小
        self.hidden_size = 768
        self.intermediate_size = 4*self.hidden_size                # MLP层两个dense层中间的intermediate state大小
        self.num_attention_heads = 12
        self.num_hidden_layers = 1 
        self.att_head_size = int(self.hidden_size/self.num_attention_heads)

        # 词表
        self.type_vocab_size = 2
        self.vocab_size = 21128

        ''' Distributed Configuration '''
        self.init_method = "tcp://192.168.124.4:23000"                         # torch.dist.init_process_group中使用的master device    
        self.distributed_backend = "gloo"
        
        # lora
        self.use_lora = False
        self.lora_att_dim = 4
        self.lora_alpha = 32
        self.lora_dropout = 0.1
        self.fan_in_fan_out = True
        self.merge_weights = False
    
    def load_from_json(self,config_file):
        if not os.path.exists(config_file):
            raise FileNotFoundError("config file: {} not found".format(config_file))
        with open(config_file, "r") as f:
            config_dict = json.load(f)
        # Data Configuration
        self.pad_size = config_dict["pad_size"]
        self.train_path = config_dict["train_path"]
        self.dev_path = config_dict["dev_path"]
        self.test_path = config_dict["test_path"]
        self.vocab_path = config_dict["vocab_path"]
        # Training Configuration
        self.device = config_dict["device"]
        if self.device == "cuda":
            self.device =  torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
        self.num_epochs = config_dict["num_epochs"]
        self.batch_size = config_dict["batch_size"]
        self.pad_size = config_dict["pad_size"]
        self.learning_rate = config_dict["learning_rate"]
        self.class_list = [x.strip() for x in open(
            "dataset/THUCNews/data/class.txt").readlines()]  
        self.num_classes = len(self.class_list)
        # Bert Configuration
        # 模型参数
        self.attention_probs_dropout_prob = config_dict["attention_probs_dropout_prob"]
        self.directionality = config_dict["directionality"]
        self.hidden_act = config_dict["hidden_act"]
        self.hidden_dropout_prob = config_dict["hidden_dropout_prob"]
        self.initializer_range = config_dict["initializer_range"]
        self.layer_norm_eps = config_dict["layer_norm_eps"]
        self.max_position_embeddings = config_dict["max_position_embeddings"]
        self.model_type = config_dict["model_type"]
        self.pad_token_id = config_dict["pad_token_id"]
        
        # 模型大小
        self.hidden_size = config_dict["hidden_size"]
        self.intermediate_size = config_dict["intermediate_size"]
        self.num_attention_heads = config_dict["num_attention_heads"]  
        self.num_hidden_layers = config_dict["num_hidden_layers"]
        self.att_head_size = int(self.hidden_size/self.num_attention_heads)
        
        #词表
        self.type_vocab_size = config_dict["type_vocab_size"]
        self.vocab_size = config_dict["vocab_size"]
        
        # Distributed Configuration
        self.init_method = config_dict["init_method"]                       # torch.dist.init_process_group中使用的master device
        self.distributed_backend = config_dict["distributed_backend"] # 通信后端
        
        # Lora
        self.use_lora = config_dict["use_lora"]
        self.lora_att_dim = config_dict["lora_att_dim"]
        self.lora_alpha = config_dict["lora_alpha"]
        self.lora_dropout = config_dict["lora_dropout"]
        self.fan_in_fan_out = config_dict["fan_in_fan_out"]
        self.merge_weights = config_dict["merge_weights"]
    
    def print_config(self):
        for k,v in self.__dict__.items():
            print(k,v)
        
config = BertConfig()
