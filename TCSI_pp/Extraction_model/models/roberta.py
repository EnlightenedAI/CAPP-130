# coding: UTF-8
import torch
import torch.nn as nn
# from pytorch_pretrained_bert import BertModel, BertTokenizer
from pytorch_pretrained import BertModel, BertTokenizer
# from transformers import BertModel, BertTokenizer
import datetime
class Config(object):

    """配置参数"""
    def __init__(self, dataset):
        self.model_name = 'roberta'
        self.train_path = dataset + '/train.txt'                                # 训练集
        self.dev_path = dataset + '/dev.txt'                                    # 验证集
        self.test_path = dataset + '/test.txt'                                  # 测试集
        self.class_list = [x.strip() for x in open(
            './TCSI_pp_zh/'+dataset + '/binary_name.txt').readlines()]                                # 类别名单
        self.save_path = dataset + self.model_name + f'_{datetime.date.today()}.ckpt'        # 模型训练结果
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备
        self.require_improvement = 2000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数
        self.num_epochs = 30                                             # epoch数
        self.batch_size = 128                                           # mini-batch大小
        self.pad_size = 150                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 5e-6                                       # 学习率
        self.roberta_path = 'TCSI_pp/Extraction_model/roberta_pretrain'
        self.tokenizer = BertTokenizer.from_pretrained(self.roberta_path)
        print(self.tokenizer)
        self.hidden_size = 768


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.roberta_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x):
        context = x[0]  # 输入的句子
        mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        _, pooled = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        out = self.fc(pooled)
        return out
