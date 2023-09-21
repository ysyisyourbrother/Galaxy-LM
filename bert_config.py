import torch

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
        self.num_epochs = 3                                             # epoch数
        self.batch_size = 10                                           # mini-batch大小
        self.pad_size = 32                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 5e-5       
        self.class_list = [x.strip() for x in open(
            "dataset/THUCNews/data/class.txt").readlines()]                                # 类别名单
        self.num_classes = len(self.class_list)                         # 类别数

        ''' Distributed Configuration '''

        ''' Bert Configuration '''
        self.attention_probs_dropout_prob = 0.1
        self.directionality = "bidi"
        self.hidden_act = "gelu"
        self.hidden_dropout_prob = 0.1
        self.hidden_size = 768
        self.initializer_range = 0.02
        self.intermediate_size = 3072                               # MLP层两个dense层中间的intermediate state大小
        self.layer_norm_eps = 1e-12
        self.max_position_embeddings = 512
        self.model_type = "bert"
        self.num_attention_heads = 12
        self.num_hidden_layers = 12
        self.pad_token_id = 0
        self.pooler_fc_size = 768
        self.pooler_num_attention_heads = 12
        self.pooler_num_fc_layers =  3
        self.pooler_size_per_head = 128
        self.pooler_type = "first_token_transform"
        self.type_vocab_size = 2
        self.vocab_size = 21128

config = BertConfig()