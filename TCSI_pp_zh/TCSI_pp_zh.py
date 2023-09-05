# coding: UTF-8

import time
import copy
import numpy as np
from  utils.test_eval import train, init_network,test as test
from utils.text_train_multi_eval import test as multi_test
from importlib import import_module
import argparse
from utils.test_utils import build_dataset, build_iterator, get_time_dif
from torch.utils.data import TensorDataset, DataLoader
import torch
import json
import os
import sys
sys.path.append('./TCSI_pp/Extraction_model/models')
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from utils.Data_loading_only_test import dataload as dataload_rewrite
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torch.utils.data import Dataset
import pytorch_lightning as pl
from transformers import (
    MT5ForConditionalGeneration,
    MT5Tokenizer,
    MT5Config,
)
parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--binary_model', type=str, default='roberta', help='choose a model: roberta, bert, ERNIE, sbert, mbert, pert')
parser.add_argument('--multi_model', type=str, default='roberta_multi', help='choose a model: roberta_multi, bert_multi, ERNIE_multi, sbert_multi, mbert_multi, pert_multi')
parser.add_argument('--data', type=str, default='test_data', help='choose a Data')
parser.add_argument('--topic_list', type=list, default=[0], help="choose a sublist from the list [0,1,2,3,4,5,6,7,8,9,10].")
parser.add_argument('--rewrite_model', type=str, default='google/mt5-small', help='choose a model: google/mt5-small')
args = parser.parse_args()

def dataload(config, data_paths):
    test_data = build_dataset(config, data_paths)
    test_sencens = []
    test_lens = []
    test_mask = []
    test_text = []
    test_id = []
    for i, test_s in enumerate(test_data):
        test_sencens.append(test_s[0])
        test_lens.append(test_s[1])
        test_mask.append(test_s[2])
        test_text.append(test_s[3])
        test_id.append(i)
    print(len(test_id))
    test_dataset = TensorDataset(torch.tensor(test_sencens), torch.tensor(test_lens), torch.tensor(test_mask),
                                 torch.tensor(test_id))
    text_iter= DataLoader(test_dataset, batch_size=config.batch_size)
    return text_iter,test_text

pl.seed_everything(42)
model_name =args.rewrite_model
tokenizer = MT5Tokenizer.from_pretrained(model_name)  # 'google/mt5-small
t5_model = MT5ForConditionalGeneration.from_pretrained(model_name)  # 'google/mt5-small'
t5_config = MT5Config.from_pretrained(model_name)
t5_model=t5_model.to(device)
t5_model.config.eos_token_id = tokenizer.eos_token_id
t5_model.config.pad_token_id = tokenizer.pad_token_id

class FalseGenerationDataset(Dataset):
    def __init__(self, tokenizer, tf_list, max_len_inp=400, max_len_out=100,templates_is_ture=True):
        self.true_false_adjective_tuples = tf_list
        self.max_len_input = max_len_inp
        self.max_len_output = max_len_out
        self.tokenizer = tokenizer
        self.inputs = []
        self.templates=[]
        self.targets = []
        self.skippedcount = 0
        self.templates_is_ture = templates_is_ture
        self._build()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()
        src_mask = self.inputs[index]["attention_mask"].squeeze()  # might need to squeeze
        target_mask = self.targets[index]["attention_mask"].squeeze()  # might need to squeeze
        labels = copy.deepcopy(target_ids)
        labels[labels == 0] = -100
        return {"source_ids": source_ids, "source_mask": src_mask,  "target_ids": target_ids, "target_mask": target_mask,
                "labels": labels}
    def _build(self):
        for inputs, outputs in self.true_false_adjective_tuples:
            input_sent = "summarization: " + inputs
            ouput_sent = outputs
            tokenized_inputs = self.tokenizer.batch_encode_plus(
                [input_sent], max_length=self.max_len_input, pad_to_max_length=True,return_tensors="pt"
            )
            tokenized_targets = self.tokenizer.batch_encode_plus(
                [ouput_sent], max_length=self.max_len_output, pad_to_max_length=True,return_tensors="pt"
            )
            self.inputs.append(tokenized_inputs)
            self.targets.append(tokenized_targets)

class T5FineTuner(pl.LightningModule):
    def __init__(self):
        super(T5FineTuner, self).__init__()
        self.model = t5_model
        self.tokenizer = tokenizer
    def forward(self, input_ids, attention_mask=None,template_ids=None,
            template_mask=None, decoder_input_ids=None, decoder_attention_mask=None,
                    lm_labels=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=lm_labels,
        )
        return outputs

if __name__ == '__main__':
    dataset=args.data
    model_name1 = args.binary_model
    model_name2 = args.multi_model
    type_list=args.topic_list
    step_1 = import_module(model_name1)
    step_2 = import_module(model_name2)
    config1 = step_1.Config(dataset)
    config2 = step_2.Config(dataset)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    start_time = time.time()

    #loading data
    print("loading data")
    data_paths='./TCSI_pp_zh/' + dataset + '/test.txt'
    test_iter,test_text =dataload(config1,data_paths)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    #important Identification
    important_model=step_1.Model(config1).to(config1.device)
    config1.important_load_path ='./TCSI_pp_zh/models/CAPP_130_roberta_pretrain/important_Identification_roberta.ckpt'
    print('loading important model')
    important_model.load_state_dict(torch.load(config1.important_load_path))
    important_model.eval()
    print('run important model')
    predict_all,pr_id=test(config1, important_model, test_iter)
    with open('./TCSI_pp_zh/' + dataset +'/test_out_text.txt','w',encoding='utf-8') as f:
        for im_label,in_text in zip(predict_all,test_text):
            if im_label == 1:
                f.write(f'{in_text}'+'\n')

    #Classification Identification
    data_paths ='./TCSI_pp_zh/' + dataset +'/test_out_text.txt'
    step_2_test_iter, step_2_test_text = dataload(config2, data_paths)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)
    Classification = step_2.Model(config2).to(config2.device)
    config2.Classification_load_path = 'TCSI_pp_zh/models/CAPP_130_roberta_pretrain/multi_classification_roberta.ckpt'
    print('loading Classification model')
    Classification.load_state_dict(torch.load(config2.Classification_load_path))
    Classification.eval()
    print('run Classification model')
    predict_all, pr_id = multi_test(config2, Classification, step_2_test_iter)
    print(predict_all)

    #Risk Identification
    print('loading risk model')
    risk_model = step_1.Model(config1).to(config1.device)  # x.Model构建模型，config.device=”cpu“
    config1.risk_load_path = 'TCSI_pp_zh/models/CAPP_130_roberta_pretrain/risk_identification_roberta.ckpt'
    risk_model.load_state_dict(torch.load(config1.risk_load_path))
    risk_model.eval()
    print('run risk model')
    predict_risk_all, pr_id = test(config1, risk_model, step_2_test_iter)
    print('id', pr_id)
    print(len(pr_id), len(predict_all), len(test_text))
    for risk_label, in_text in zip(predict_risk_all, step_2_test_text):
        if risk_label == 1:
            print('risk-sentence', in_text)
    print(predict_risk_all)
    with open('./TCSI_pp_zh/' + dataset +'/test_out_text.json', 'w', encoding='utf-8') as f:
        for label, in_text, risk_label in zip(predict_all, step_2_test_text, predict_risk_all):
            print(risk_label)
            f.write(json.dumps({'text': in_text, 'label': int(label), 'highlight': int(risk_label)},
                               ensure_ascii=False) + '\n')

    #rewrite
    data_name = './TCSI_pp_zh/' + dataset +'/test_out_text.json'
    load_path = "./TCSI_pp_zh/models/CAPP_130_mt5_pretrain/mT5_small.ckpt"
    true_false_adjective_tuples_test = dataload_rewrite(data_name)
    save_rewrite_path = './TCSI_pp_zh/' + dataset +'/test_rewrite_sentences.json'  # 保存结果
    new_model = T5FineTuner().to(device)
    new_model = new_model.load_from_checkpoint(checkpoint_path=load_path)
    new_model.to(device)
    new_model.eval()
    new_model.model.to(device)
    with open(save_rewrite_path, 'w', encoding='utf-8') as fp:
        for text in true_false_adjective_tuples_test:
            test_tokenized = tokenizer.encode_plus('summarization' + text[0][:350], return_tensors="pt")
            test_input_ids = test_tokenized["input_ids"].to(device)
            test_attention_mask = test_tokenized["attention_mask"].to(device)
            beam_outputs = new_model.model.generate(
                input_ids=test_input_ids,
                attention_mask=test_attention_mask,
                max_length=250,
                early_stopping=True,
                num_beams=10,
                num_return_sequences=1,
                no_repeat_ngram_size=2,
            )

            for beam_output in beam_outputs:
                sent = tokenizer.decode(beam_output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            sent = ''.join(sent.split())
            fp.write(json.dumps({'text': text[0], 'rewrite': sent, "label": text[1], 'highlight': text[2]},
                                ensure_ascii=False) + '\n')
    write_into_pp=''
    type_list=args.type_list
    type_title=['我们采集的信息与数据',
                '我们获取的应用权限',
                '我们向第三方共享、转让，委托第三方处理及公开披露的数据',
                '如何使用您的数据',
                '我们如何存储您的数据',
                '我们会采用哪些措施保护您的数据',
                '针对特殊人群的条款',
                '您可以如何管理您的数据或取消授权',
                '如何与我们取得联系',
                '隐私政策的授权与变更',
                '停止运营后我们会如何处理您的数据'
                ]
    f=open(save_rewrite_path,'r',encoding='utf-8').readlines()
    for i,type_i in enumerate(type_list):
        write_into_pp += f"<h3>{i+1}.{type_title[type_i]}</h3>\n"
        j=1
        for sen in f:
            sen=json.loads(sen)
            if int(sen['label'])==int(type_i):
                if sen['highlight']==0:
                    write_into_pp += f"<p>（{j}）{sen['rewrite']}</p>\n"
                else:
                    write_into_pp += f'<p style="font-family: Arial, sans-serif; color: red;">（{j}）<b>{sen["rewrite"]}</b></p>\n'
                j+=1
        if j==1:
            write_into_pp += f'<p style="font-family: Arial, sans-serif; color: red;"><b>没有找到与“{type_title[type_i]}“相关的条款</b></p>\n'
    html_content=f"""
    <!DOCTYPE html>
    <html>
    <head>
    <title>隐私政策摘要</title>
    <style>
    body {{
      display: flex;
      flex-direction: column;
      height: 100vh;
      text-align: left;
      margin: 0;
      padding: 0;
      box-sizing: border-box;
    }}
    .container {{
      width: 80%;
      max-width: 800px;
      margin: 0 auto;
    }}
  </style>
      
    </head>
    <body>
    <div class="container">
    <h1>隐私政策摘要</h1>
    {write_into_pp}
    </div>
    </body>
    </html>
    """

    # Write summarition to HTML.
    with open(f'./TCSI_pp_zh/{dataset}/pp_sum.html', "w") as file:
        file.write(html_content)
    print("Summarition has been written to HTML.")