# coding: UTF-8
import torch
import torch.nn as nn
# from pytorch_pretrained_bert import BertModel, BertTokenizer
from pytorch_pretrained import BertModel, BertTokenizer

class Config(object):

    """配置参数"""
    def __init__(self):
        self.model_name = 'ERNIE_multi'
        self.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        self.require_improvement = 2000
        self.num_classes = 11
        self.num_epochs = 100
        self.batch_size = 32
        self.pad_size = 100
        self.learning_rate = 5e-6
        self.bert_path = './ERNIE_pretrain'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        print(self.tokenizer)
        self.hidden_size = 768
        self.focalloss_rate = 0.5
        self.is_model_name = 'ERNIE'

class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(config.hidden_size, config.num_classes)

        self.m = nn.Sigmoid()
    def forward(self, x):
        context = x[0]
        mask = x[2]
        _, pooled = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        out = self.fc(pooled)
        return out
