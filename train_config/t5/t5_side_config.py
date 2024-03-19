import torch
import os
import json 
from  .t5_config import T5Config
class T5SideConfig(T5Config):
    # model_type = "t5"
    # keys_to_ignore_at_inference = ["past_key_values"]
    # attribute_map = {"hidden_size": "d_model", "num_attention_heads": "num_heads", "num_hidden_layers": "num_layers"}

    def __init__(
        self
    ):
        super(T5SideConfig,self).__init__()
        # # 训练、验证、测试集数据路径
        # self.train_path = "dataset/THUCNews/data/train.txt"
        # self.dev_path = "dataset/THUCNews/data/dev.txt"
        # self.test_path = "dataset/THUCNews/data/test.txt"
        # self.vocab_path = "dataset/THUCNews/vocab.txt"
        # ''' Training Configuration '''
        # self.train = True
        # self.device = "cuda"
        # if self.device == "cuda":
        #     self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备
        # else:
        #     self.device = torch.device('cpu')
        # self.num_epochs = 3                                             # epoch数
        # self.batch_size = 4                                           # mini-batch大小
        # self.pad_size = 32                                              # 每句话处理成的长度(短填长切)
        # self.learning_rate = 5e-5       
        # self.class_list = [x.strip() for x in open(
        #     "dataset/THUCNews/data/class.txt").readlines()]                                # 类别名单
        # self.num_classes = len(self.class_list)  
        # #
        # self.type_vocab_size = 2
        # self.vocab_size = 32128 # TODO: 现在是 T5 32128 和bert不一样 
        # # 模型参数
        # self.d_model = 768
        # self.d_kv = 64
        # self.d_ff = 3072
        # self.num_layers = 12
        # self.num_decoder_layers = 12
        # self.num_heads = 12
        # self.relative_attention_num_buckets = 32
        # self.relative_attention_max_distance = 128
        # self.dropout_rate = 0.1
        # self.classifier_dropout = 0.0
        # self.layer_norm_epsilon = 1e-06
        # self.initializer_factor = 1.0
        # self.feed_forward_proj = "relu"

        # act_info = self.feed_forward_proj.split("-")
        # self.dense_act_fn = act_info[-1]
        # self.is_gated_act = act_info[0] == "gated"
        # if len(act_info) > 1 and act_info[0] != "gated" or len(act_info) > 2:
        #     raise ValueError(
        #         f"`feed_forward_proj`: {self.feed_forward_proj } is not a valid activation function of the dense layer. "
        #         "Please make sure `feed_forward_proj` is of the format `gated-{ACT_FN}` or `{ACT_FN}`, e.g. "
        #         "'gated-gelu' or 'relu'"
        #     )
        # # for backwards compatibility
        # if self.feed_forward_proj  == "gated-gelu":
        #     self.dense_act_fn = "gelu_new"
        # # fixed
        # self.decoder_start_token_id = 0
        # self.is_encoder_decoder = True
        # self.pad_token_id = 0
        # self.eos_token_id = 1
        # self.return_dict = False
        # self.use_cache = False
        # self.output_attentions = False  
        # self.output_hidden_states = False
        # side
        self.side_reduction_factor = 8
        self.add_bias_sampling = True #TODO: 是什么

    def load_from_json(self,config_file):
        super().load_from_json(config_file)
        if not os.path.exists(config_file):
            raise FileNotFoundError("config file: {} not found".format(config_file))
        with open(config_file, "r") as f:
            config_dict = json.load(f)
            print("========== Updating config from file: ", config_file,"==========")
        # """ Data Configuration """
        # self.train_path = config_dict["train_path"]
        # self.dev_path = config_dict["dev_path"]
        # self.test_path = config_dict["test_path"]
        # self.vocab_path = config_dict["vocab_path"]
        # # Training Configuration
        # self.train = config_dict["train"]
        # self.device = config_dict["device"]
        # if self.device == "cuda":
        #     self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备
        # else:
        #     self.device = torch.device('cpu')
        # self.num_epochs = config_dict["num_epochs"]
        # self.batch_size = config_dict["batch_size"]
        # self.pad_size = config_dict["pad_size"]
        # self.learning_rate = config_dict["learning_rate"]
        # self.class_list = [x.strip() for x in open(
        #     "dataset/THUCNews/data/class.txt").readlines()]  
        # self.num_classes = len(self.class_list)
        # #  模型参数
        # self.d_model = config_dict["d_model"]
        # self.d_kv =  config_dict["d_kv"]
        # self.d_ff =  config_dict["d_ff"]
        # self.num_layers =  config_dict["num_layers"]
        # self.num_decoder_layers =   config_dict["num_decoder_layers"]
        # self.num_heads =  config_dict["num_heads"]
        # self.relative_attention_num_buckets =  config_dict["relative_attention_num_buckets"]
        # self.relative_attention_max_distance =  config_dict["relative_attention_max_distance"]
        # self.dropout_rate =  config_dict["dropout_rate"]
        # self.classifier_dropout =  config_dict["classifier_dropout"] 
        # self.layer_norm_epsilon =  config_dict["layer_norm_epsilon"]
        # self.initializer_factor =   config_dict["initializer_factor"]
        # self.feed_forward_proj =  config_dict["feed_forward_proj"]
        
        # act_info = self.feed_forward_proj.split("-")
        # self.dense_act_fn = act_info[-1]
        # self.is_gated_act = act_info[0] == "gated"
        # if len(act_info) > 1 and act_info[0] != "gated" or len(act_info) > 2:
        #     raise ValueError(
        #         f"`feed_forward_proj`: {self.feed_forward_proj } is not a valid activation function of the dense layer. "
        #         "Please make sure `feed_forward_proj` is of the format `gated-{ACT_FN}` or `{ACT_FN}`, e.g. "
        #         "'gated-gelu' or 'relu'"
        #     )
        # # for backwards compatibility
        # if self.feed_forward_proj  == "gated-gelu":
        #     self.dense_act_fn = "gelu_new"
        # # fixed
        # self.decoder_start_token_id = 0
        # self.is_encoder_decoder = True
        # self.pad_token_id = 0
        # self.eos_token_id = 1
        # self.return_dict = False
        # self.use_cache = False
        # self.output_attentions = False  
        # self.output_hidden_states = False
        self.side_reduction_factor =  config_dict["side_reduction_factor"]
        self.add_bias_sampling = config_dict["add_bias_sampling"]
    def print_config(self):
        for k,v in self.__dict__.items():
            print(k,v)