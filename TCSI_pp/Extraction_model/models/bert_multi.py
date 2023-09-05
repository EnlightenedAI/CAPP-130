# coding: UTF-8
import torch
import torch.nn as nn
from pytorch_pretrained import BertModel,BertTokenizer

class Config(object):
    """配置参数"""
    def __init__(self):
        self.model_name = 'bert_multi'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.require_improvement = 2000
        self.num_classes =11
        self.num_epochs = 100
        self.batch_size = 64
        self.pad_size =100
        self.learning_rate = 5e-6
        self.bert_path = './bert_pretrain'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768
        self.focalloss_rate=0.5
        self.is_model_name='bert'
class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(config.hidden_size*1, config.num_classes)

    def forward(self, x):
        context = x[0]
        mask = x[2]
        _, pooled1 = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        out= self.fc(pooled1)
        return out