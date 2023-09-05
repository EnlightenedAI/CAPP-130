# coding: UTF-8
import torch
import torch.nn as nn
# from pytorch_pretrained_bert import BertModel, BertTokenizer
from pytorch_pretrained import BertModel, BertTokenizer
from transformers import AutoTokenizer,AutoModel
import datetime

class Config(object):

    """配置参数"""
    def __init__(self, dataset):
        self.model_name = 'sbert_multi'
        self.train_path = dataset + '/train.txt'                                # 训练集
        self.dev_path = dataset + '/dev.txt'                                    # 验证集
        self.test_path = dataset + '/test.txt'                                  # 测试集
        self.class_list = [x.strip() for x in open(
            dataset + '/../class.txt').readlines()]                                # 类别名单
        self.save_path = dataset + self.model_name + f'_{datetime.date.today()}.ckpt'        # 模型训练结果
        self.save_path = dataset + self.model_name + f'_{datetime.date.today()}.ckpt'  # 模型训练结果
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = 11#len(self.class_list)                         # 类别数
        self.num_epochs = 30                                             # epoch数
        self.batch_size = 128                                           # mini-batch大小
        self.pad_size = 150#150                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 2e-6                                       # 学习率
        self.bert_path = 'uer/sbert-base-chinese-nli'
        self.tokenizer = AutoTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = AutoModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x):
        context = x[0]  # 输入的句子
        mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        outputs = self.bert(context, attention_mask=mask)
        pooled = torch.mean(outputs['last_hidden_state'], dim=1)  # 从输出中获取 pooled representation

        out = self.fc(pooled)
        return out