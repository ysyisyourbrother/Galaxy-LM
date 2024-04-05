
import warnings
import torch
import os
import json 


class BartConfig():
    model_type = "bart"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {"num_attention_heads": "encoder_attention_heads", "hidden_size": "d_model"}

    def __init__(
        self,
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
        self.batch_size = 4                                           # mini-batch大小
        self.pad_size = 32                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 5e-5       
        self.class_list = [x.strip() for x in open(
            "dataset/THUCNews/data/class.txt").readlines()]                                # 类别名单
        self.num_classes = len(self.class_list)  
        ## 模型参数
        self.type_vocab_size = 2
        self.vocab_size = 50265 # T
        self.dropout =  0.1
        self.attention_dropout = 0.1
        self.activation_dropout = 0.1
        self.activation_function = "gelu"
        self.init_std = 0.02
        self.encoder_layerdrop = 0.0
        self.decoder_layerdrop = 0.0
        self.classifier_dropout = 0.0
        self.use_cache = False
        self.return_dict = False
        self.output_attentions = False  
        self.output_hidden_states = False
        self.scale_embedding = False  # scale factor will be sqrt(d_model) if True
        self.num_labels=3
        self.pad_token_id=1
        self.bos_token_id=0
        self.eos_token_id=2
        self.is_encoder_decoder=True
        self.decoder_start_token_id=0
        self.forced_eos_token_id=2
        ##################################################
        # 模型大小参数
        self.max_position_embeddings = 1024
        self.d_model = 1024
        self.encoder_ffn_dim = 4096
        self.encoder_layers = 12
        self.encoder_attention_heads =  16
        self.decoder_ffn_dim = 4096
        self.decoder_layers = 12
        self.decoder_attention_heads = 16
        self.num_hidden_layers = 12
        ######################################################
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
        self.check_adapter_config()
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
    def print_config(self):
        for k,v in self.__dict__.items():
            print(k,v)
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