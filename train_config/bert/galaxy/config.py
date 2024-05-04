import torch
import os
import json 
class BertConfig():
    def __init__(self):
        ''' Data Configuration '''
        # 训练、验证、测试集数据路径
        self.train_path = "dataset/THUCNews/data/train.txt"
        self.dev_path = "dataset/THUCNews/data/dev.txt"
        self.test_path = "dataset/THUCNews/data/test.txt"
        self.vocab_path = "dataset/THUCNews/vocab.txt"

        ''' Training Configuration '''
        self.train = True
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备
        self.num_epochs = 3                                             # epoch数
        self.batch_size = 4                                           # mini-batch大小
        self.pad_size = 32                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 5e-5       
        self.class_list = [x.strip() for x in open(
            "dataset/THUCNews/data/class.txt").readlines()]                                # 类别名单
        self.num_classes = len(self.class_list)                         # 类别数
        # 词表
        self.type_vocab_size = 2
        self.vocab_size = 21128
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
        self.num_hidden_layers = 2
        self.att_head_size = int(self.hidden_size/self.num_attention_heads)
        # full model
        self.full_model = True
        # lora 
        self.use_lora = False
        self.lora_dim = 4
        self.lora_alpha = 32
        self.lora_dropout = 0.1
        self.fan_in_fan_out = True
        self.merge_weights = False
        # adapter
        self.use_adapter = False
        self.adapter_reduction_dim = 16
        self.non_linearity = "gelu_new"
         # side
        self.use_side = False
        self.side_reduction_factor = 8
        self.add_bias_sampling = True
        # side-only, forward free
        self.use_side_only = False
        self.check_adapter_config()

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
        if not os.path.exists(config_file):
            raise FileNotFoundError("config file: {} not found".format(config_file))
        with open(config_file, "r") as f:
            config_dict = json.load(f)
            print("========== Updating config from file: ", config_file,"==========")  
            for key, value in config_dict.items():
                if hasattr(self, key):
                    setattr(self, key, value)
                else:
                    print(f"Ignoring unknown attribute: {key}")
        self.check_adapter_config()
        
        if (sum(self.tp_num_attention_heads_list) != self.num_attention_heads):
            raise ValueError("Sum of tp_num_attention_heads_list must equal to num_attention_heads")
        if (sum(self.tp_intermediate_size_list) != self.intermediate_size):
            raise ValueError("Sum of tp_intermediate_size_list must equal to intermediate_size")
    def print_config(self):
        for k,v in self.__dict__.items():
            print(k,v)
    def update_galaxy_config(self, args):
        rank = args.rank
        world = args.world
        self.tp_num_attention_heads = self.tp_num_attention_heads_list[rank]
        self.tp_intermediate_size = self.tp_intermediate_size_list[rank]
    def check_adapter_config(self):
        if self.use_lora :
            assert self.lora_dim > 0
        if self.use_adapter:
            assert self.adapter_reduction_dim > 0
        if self.use_side or self.use_side_only:
            assert self.side_reduction_factor > 0
        #full / lora / adapter / side 只能一个true
        true_count = sum([1 for value in [self.use_lora, 
                                      self.use_adapter,
                                      self.use_side,
                                      self.use_side_only,
                                      self.full_model] if value])
        assert true_count == 1
        print("========== check adapter config ==========")
        if self.use_lora:
            print("use lora")
        if self.use_adapter:
            print("use adapter")
        if self.use_side:
            print("use side")
        if self.full_model:
            print("full model")
        if self.use_side_only:
            print("use side only")
        print("==========                       ==========")