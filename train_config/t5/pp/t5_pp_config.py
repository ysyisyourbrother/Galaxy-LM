import torch
import os
import json 
from transformers.configuration_utils import PretrainedConfig

class T5PPConfig(PretrainedConfig ):
    # model_type = "t5"
    # keys_to_ignore_at_inference = ["past_key_values"]
    # attribute_map = {"hidden_size": "d_model", "num_attention_heads": "num_heads", "num_hidden_layers": "num_layers"}

    def __init__(
        self
    ):
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
        self.batch_size = 4                                              # mini-batch大小
        self.pad_size = 128                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 5e-5       
        self.class_list = [x.strip() for x in open(
            "dataset/THUCNews/data/class.txt").readlines()]                                # 类别名单
        self.num_classes = len(self.class_list)  
        #
        self.type_vocab_size = 2
        self.vocab_size = 32128 # TODO: 现在是 T5 32128 和bert不一样 
        # 模型参数
        self.d_model = 768
        self.d_kv = 64
        self.d_ff = 3072
        self.num_layers = 12
        self.num_decoder_layers = 12
        self.num_heads = 12
        self.relative_attention_num_buckets = 32
        self.relative_attention_max_distance = 128
        self.dropout_rate = 0.1
        self.classifier_dropout = 0.0
        self.layer_norm_epsilon = 1e-06
        self.initializer_factor = 1.0
        self.feed_forward_proj = "relu"

        act_info = self.feed_forward_proj.split("-")
        self.dense_act_fn = act_info[-1]
        self.is_gated_act = act_info[0] == "gated"
        if len(act_info) > 1 and act_info[0] != "gated" or len(act_info) > 2:
            raise ValueError(
                f"`feed_forward_proj`: {self.feed_forward_proj } is not a valid activation function of the dense layer. "
                "Please make sure `feed_forward_proj` is of the format `gated-{ACT_FN}` or `{ACT_FN}`, e.g. "
                "'gated-gelu' or 'relu'"
            )
        # for backwards compatibility
        if self.feed_forward_proj  == "gated-gelu":
            self.dense_act_fn = "gelu_new"
        # fixed
        self.decoder_start_token_id = 0
        self.is_encoder_decoder = True
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.return_dict = False
        self.use_cache = False
        self.output_attentions = False  
        self.output_hidden_states = False
        ###########################################
        # full model
        self.full_model = True
        # lora
        self.use_lora = False
        self.lora_dim = 32
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
        
        # TODO:side only
        self.check_adapter_config()
        # Distributed Configuration
        self.init_method = "tcp://127.0.0.1:23000"                         # torch.dist.init_process_group中使用的master device    
        self.distributed_backend = "gloo"
        self.stage_num_hidden_layers_list = [6,6,6,6] 
        self.num_microbatches = 2
        self.num_iterations = 1
        #  Pipeline Configuration
        self.stage = None
        self.total_stage = None
        self.pre_rank = None
        self.next_rank = None
        self.is_first_stage = None
        self.is_last_stage = None
        self.left_layer_index = None
        self.right_layer_index = None
        self.num_pp_encoder_layers = None
        self.num_pp_decoder_layers = None
        self.is_encoder = None
        self.is_encoder_first = None
        self.is_encoder_last = None
        self.is_decoder = None
        self.is_decoder_first = None
        self.is_decoder_last = None
    
    def check_pp_config(self):
        if self.total_stage==1:
            raise ValueError("total_stage should > 1")
        if self.total_stage != len(self.stage_num_hidden_layers_list):
            raise ValueError("total_stage != len(stage_num_hidden_layers_list)")
        if sum(self.stage_num_hidden_layers_list) != self.num_layers +  self.num_decoder_layers:
            raise ValueError("sum of stage_hidden_layers_num_list should be equal to num_hidden_layers")
    
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
        #  模型参数
        self.d_model = config_dict["d_model"]
        self.d_kv =  config_dict["d_kv"]
        self.d_ff =  config_dict["d_ff"]
        self.num_layers =  config_dict["num_layers"]
        self.num_decoder_layers =   config_dict["num_decoder_layers"]
        self.num_heads =  config_dict["num_heads"]
        self.relative_attention_num_buckets =  config_dict["relative_attention_num_buckets"]
        self.relative_attention_max_distance =  config_dict["relative_attention_max_distance"]
        self.dropout_rate =  config_dict["dropout_rate"]
        self.classifier_dropout =  config_dict["classifier_dropout"] 
        self.layer_norm_epsilon =  config_dict["layer_norm_epsilon"]
        self.initializer_factor =   config_dict["initializer_factor"]
        self.feed_forward_proj =  config_dict["feed_forward_proj"]
        
        act_info = self.feed_forward_proj.split("-")
        self.dense_act_fn = act_info[-1]
        self.is_gated_act = act_info[0] == "gated"
        if len(act_info) > 1 and act_info[0] != "gated" or len(act_info) > 2:
            raise ValueError(
                f"`feed_forward_proj`: {self.feed_forward_proj } is not a valid activation function of the dense layer. "
                "Please make sure `feed_forward_proj` is of the format `gated-{ACT_FN}` or `{ACT_FN}`, e.g. "
                "'gated-gelu' or 'relu'"
            )
        # for backwards compatibility
        if self.feed_forward_proj  == "gated-gelu":
            self.dense_act_fn = "gelu_new"
        # fixed
        self.decoder_start_token_id = 0
        self.is_encoder_decoder = True
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.return_dict = False
        self.use_cache = False
        self.output_attentions = False  
        self.output_hidden_states = False
        ########################################################### 
        # full model
        self.full_model = config_dict["full_model"]
        # Lora
        self.use_lora = config_dict["use_lora"]
        if self.use_lora:
            self.lora_dim = config_dict["lora_dim"]
            self.lora_alpha = config_dict["lora_alpha"]
            self.lora_dropout = config_dict["lora_dropout"]
            self.fan_in_fan_out = config_dict["fan_in_fan_out"]
            self.merge_weights = config_dict["merge_weights"]
        # Adapter
        self.use_adapter = config_dict["use_adapter"]
        if self.use_adapter:
            self.adapter_reduction_dim = config_dict["adapter_reduction_dim"]
            self.non_linearity = config_dict["non_linearity"]
        # side
        self.use_side =  config_dict["use_side"]
        if self.use_side:
            self.side_reduction_factor =  config_dict["side_reduction_factor"]
            self.add_bias_sampling =  config_dict["add_bias_sampling"]
        # side only:
        self.use_side_only =  config_dict["use_side_only"]
        if self.use_side_only:
            self.side_reduction_factor =  config_dict["side_reduction_factor"]
            self.add_bias_sampling =  config_dict["add_bias_sampling"]
        self.check_adapter_config()
        ##############################################################
        # Distributed Configuration
        self.init_method = config_dict["init_method"]                       # torch.dist.init_process_group中使用的master device
        self.distributed_backend = config_dict["distributed_backend"] # 通信后端
        self.stage_num_hidden_layers_list = config_dict["stage_num_hidden_layers_list"]
        self.num_microbatches = config_dict["num_microbatches"]
        self.num_iterations = config_dict["num_iterations"]
        
    def print_config(self):
        for k,v in self.__dict__.items():
            print(k,v)
    def update_pp_stage_config(self, args):
        self.stage = args.rank
        self.total_stage =args.world 
        self.pre_rank = None if self.stage == 0 else self.stage - 1
        self.next_rank = None if self.stage  == self.total_stage -1 else self.stage + 1
        self.is_first_stage = (self.stage ==0)  # 第一个需要过embedding
        self.is_last_stage =  (self.stage  ==  self.total_stage - 1)   #最后一层需要过 和lm_head
        
        self.left_layer_index = sum(self.stage_num_hidden_layers_list[:self.stage  ]) + 1
        self.right_layer_index = sum(self.stage_num_hidden_layers_list[:self.stage + 1])
        if  self.right_layer_index <= self.num_layers:
            self.num_pp_encoder_layers = self.right_layer_index  - self.left_layer_index + 1
            self.num_pp_decoder_layers = 0
        elif   self.left_layer_index <=  self.num_layers:
            self.num_pp_encoder_layers =  self.num_layers - self.left_layer_index + 1
            self.num_pp_decoder_layers = self.right_layer_index - (self.num_layers+1) + 1
        else:
            self.num_pp_encoder_layers = 0
            self.num_pp_decoder_layers =  self.right_layer_index  - self.left_layer_index + 1
        self.is_encoder = self.num_pp_encoder_layers > 0
        self.is_encoder_first = (self.left_layer_index == 1)
        self.is_encoder_last = (self.right_layer_index == self.num_layers)
        self.is_decoder = self.num_pp_decoder_layers > 0
        self.is_decoder_first = (self.left_layer_index == self.num_layers + 1 )
        self.is_decoder_last = (self.right_layer_index == self.num_layers * 2)
        if self.is_encoder and self.is_decoder:
            raise ValueError("Not support encoder and decoder at the same time")
        print("== update PP stage config \n== stage={}\n, total_stage={}\n, left index={}, right index={}, num_pp_encoder_layers={}, num_pp_decoder_layers={}".format(self.stage, self.total_stage, self.left_layer_index, self.right_layer_index, self.num_pp_encoder_layers, self.num_pp_decoder_layers))
        self.check_pp_config()