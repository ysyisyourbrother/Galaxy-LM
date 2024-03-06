import torch
import os
import json 

class PPLlamaConfig():
    def __init__(self):
        ''' Data Configuration '''
        # 训练、验证、测试集数据路径
        self.train_path = "dataset/THUCNews/data/train.txt"
        self.dev_path = "dataset/THUCNews/data/dev.txt"
        self.test_path = "dataset/THUCNews/data/test.txt"
        self.vocab_path = "dataset/THUCNews/vocab.txt"
        
        ''' Training Configuration '''
        self.train = True
        self.device = "cuda"
        if self.device == "cuda":
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备
        else:
            self.device = torch.device('cpu')
        self.num_epochs = 3                                             # epoch数
        self.batch_size =4                                           # mini-batch大小
        self.pad_size = 32                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 5e-5       
        self.class_list = [x.strip() for x in open(
            "dataset/THUCNews/data/class.txt").readlines()]                                # 类别名单
        self.num_classes = len(self.class_list)  
        #  模型参数
        self.hidden_act  = "silu"
        self.initializer_range = 0.02
        self.rms_norm_eps=1e-6
        self.use_cache = False
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.tie_word_embeddings = False
        self.pad_token_id = 0
        self.max_position_embeddings = 512
        self.model_type = "llama"
        # model 大小 
        self.hidden_size = 4096
        self.intermediate_size = 11008
        self.num_hidden_layers = 2
        self.num_attention_heads = 32
        self.att_head_size = int(self.hidden_size/self.num_attention_heads)
        # 词表
        self.type_vocab_size = 2
        self.vocab_size = 21128
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
            print("========== Updating config from file: ", config_file,"==========")
            """ Data Configuration """
        self.train_path = config_dict["train_path"]
        self.dev_path = config_dict["dev_path"]
        self.test_path = config_dict["test_path"]
        self.vocab_path = config_dict["vocab_path"]
            # Training Configuration
        self.train = config_dict["train"]
        self.device = config_dict["device"]
        if self.device == "cuda":
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备
        else:
            self.device = torch.device('cpu')
        self.num_epochs = config_dict["num_epochs"]
        self.batch_size = config_dict["batch_size"]
        self.pad_size = config_dict["pad_size"]
        self.learning_rate = config_dict["learning_rate"]
        self.class_list = [x.strip() for x in open(
            "dataset/THUCNews/data/class.txt").readlines()]  
        self.num_classes = len(self.class_list)
        
        # 模型参数
        self.hidden_act=config_dict["hidden_act"]
        self.initializer_range = config_dict["initializer_range"]
        self.rms_norm_eps=config_dict["rms_norm_eps"]
        self.use_cache=config_dict["use_cache"]
        self.pad_token_id = config_dict["pad_token_id"]
        self.bos_token_id = config_dict["bos_token_id"]
        self.eos_token_id = config_dict["eos_token_id"]
        self.tie_word_embeddings = config_dict["tie_word_embeddings"]
        self.pad_token_id = config_dict["pad_token_id"]
        self.max_position_embeddings = config_dict["max_position_embeddings"]
        self.model_type = config_dict["model_type"]
        # model 大小 
        self.hidden_size = config_dict["hidden_size"]
        self.intermediate_size = config_dict["intermediate_size"]
        self.num_hidden_layers = config_dict["num_hidden_layers"]
        self.num_attention_heads = config_dict["num_attention_heads"]
        self.att_head_size = int(self.hidden_size/self.num_attention_heads)
        #词表
        self.type_vocab_size = config_dict["type_vocab_size"]
        self.vocab_size = config_dict["vocab_size"]
        #  Distributed Configuration
        self.init_method = config_dict["init_method"]                       # torch.dist.init_process_group中使用的master device
        self.distributed_backend = config_dict["distributed_backend"] # 通信后端
        self.stage_num_hidden_layers_list = config_dict["stage_num_hidden_layers_list"]
        self.num_microbatches = config_dict["num_microbatches"]
        self.num_iterations = config_dict["num_iterations"]
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
        self.is_first_stage = (self.stage ==0)  # 第一个需要过embedding
        self.is_last_stage =  (self.stage  ==  self.total_stage - 1)   #最后一层需要过 pool和lm_head
        print("="*20)
        print("== update PP stage config \n== stage={}\n, total_stage={}\n, num_pp_hidden_layers={}".format(self.stage, self.total_stage, self.num_pp_hidden_layers))
        print("="*20)
 