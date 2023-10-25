# coding: UTF-8
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
from utils import get_time_dif
from pytorch_pretrained.optimization import BertAdam
from sklearn.metrics import recall_score,f1_score,precision_score,roc_auc_score
from loss.focalloss import BCEFocalLoss
from loss.Focal_loss import focal_loss
import os
from sklearn.metrics import precision_recall_curve
from numpy.core.fromnumeric import argmax

def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if len(w.size()) < 2:
                continue
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass


def train(config, model, train_iter, dev_iter, test_iter):
    start_time = time.time()
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=config.learning_rate,
                         warmup=0.05,
                         t_total=len(train_iter) * config.num_epochs)
    total_batch: int = 0 
    dev_best_loss = float('inf')
    last_improve = 0 
    flag = False
    model.train()
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        for i, (trains_1, labels,lens,masks) in enumerate(train_iter):
            trains=(trains_1.to(config.device),lens.to(config.device),masks.to(config.device))
            outputs = model(trains)
            model.zero_grad()
            labels=labels.to(config.device)
            criterion_weighted=focal_loss(alpha=config.focalloss_rate, gamma=2, num_classes=2)
            loss = criterion_weighted(outputs, labels)
            loss.backward()
            optimizer.step()
            if total_batch % 500 == 0:
                true = labels.data.cpu()
                predic = outputs.data.argmax(axis=1)
                predict_true=outputs.data[:,1]
                train_acc = metrics.accuracy_score(true.clone().detach().cpu(), predic.clone().detach().cpu())
                train_pre = precision_score(np.array(true.clone().detach().cpu()), np.array(predic.clone().detach().cpu()),average=None)
                train_rec = recall_score(np.array(true.clone().detach().cpu()), np.array(predic.clone().detach().cpu()),average=None)
                train_f1 = f1_score(np.array(true.clone().detach().cpu()), np.array(predic.clone().detach().cpu()),average=None)
                train_auc ="none"
                dev_acc, dev_loss,dev_auc,dev_pre,dev_rec,dev_f1= evaluate(config, model, dev_iter)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
                tra_msg='Tra_Pre: {0:>6.2%},    Tra_Rec: {1:>6.2%},    Tra_F1: {2:>6.2%},   Val_Pre: {3:>6.2%},    Val_Rec: {4:>6.2%},   Val_F1: {5:>6.2%}'
                with open(config.save_path_acc_loss,"a",encoding="utf-8") as write_loss:
                    write_loss.write(json.dumps({'Iter': str(total_batch),'Train Loss': str(loss.item()),'Train Acc': str(train_acc),
                                                 'Val Loss': str(dev_loss.item()),'Val Acc': str(dev_acc),'Time': str(time_dif),
                                                 'train_Precision': str(train_pre),'train_Recall': str(train_rec),'train_F1': str(train_f1),
                                                 'Val_Precision': str(dev_pre),'Val_Recall': str(dev_rec), 'Val_F1': str(dev_f1)},ensure_ascii=False)+ '\n')
                write_loss.close()
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
                model.train()
            total_batch += 1
            if total_batch - last_improve > config.require_improvement:
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break
    test(config, model, test_iter)


def test(config, model, test_iter):
    start_time = time.time()
    test_acc, test_loss, test_report, test_confusion ,pre,rec,f1,auc= evaluate(config, model, test_iter, test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    with open("./"+config.dataset+"/output/testput/"+config.dataset+"_"+config.is_model_name+"_focalloss_rate_"+str(config.focalloss_rate).replace(str(config.focalloss_rate)[1], "", 1)+".json", "a") as test_out:
        test_out.write(json.dumps({"test_acc":test_acc,'test_precision':pre,'test_recall':rec,'test_f1':f1,'test_auc':auc,'test_report':test_report})+'\n')
    test_out.close()
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([[]], dtype=int)
    labels_all = np.array([[]], dtype=int)
    predict_true_all = np.array([[]], dtype=int)
    
    with torch.no_grad():
        for texts, labels , lens,masks in data_iter:
            texts=(texts.to(config.device),lens.to(config.device),masks.to(config.device))
            outputs = model(texts)
            labels = labels.to(config.device)
            criterion_weighted = focal_loss(alpha=config.focalloss_rate, gamma=2, num_classes=2)
            loss = criterion_weighted(outputs, labels)
            loss_total += loss
            predic = outputs.data.argmax(axis=1)
            predict_true=outputs.data[:,1]
            labels_all = np.append(labels_all, labels.cpu().numpy())
            predict_all = np.append(predict_all, predic.cpu().numpy())
            predict_true_all=np.append(predict_true_all, predict_true.cpu().numpy())

    acc = metrics.accuracy_score(np.array(labels_all.data), np.array(predict_all))
    pre = precision_score(np.array(labels_all), np.array(predict_all),average=None)
    rec = recall_score(np.array(labels_all), np.array(predict_all),average=None)
    f1 = f1_score(np.array(labels_all), np.array(predict_all),average=None)
    auc = roc_auc_score(np.array(labels_all.data), np.array(predict_true_all))
    if test:
        pre=precision_score(np.array(labels_all), np.array(predict_all))
        rec=recall_score(np.array(labels_all), np.array(predict_all))
        f1=f1_score(np.array(labels_all), np.array(predict_all))
        report = metrics.classification_report(np.array(labels_all.data),np.array(predict_all))
        confusion = metrics.multilabel_confusion_matrix(np.array(labels_all.data), np.array(predict_all))
        c=labels_all==predict_all
        with open('error/predict_label.txt', 'w') as fs:
            for s in predict_all:
                fs.write(str(s))
                fs.write('\n')
        fs.close()
        x = []
        with open('error/summarization.txt', 'w') as fs:
            for s in c:
                fs.write(str(s))
                fs.write('\n')
        fs.close()
        x = []
        with open('error/error.txt', 'w', encoding='UTF-8') as fs2:
            with open('concat_data/data/class12/train.txt', 'r', encoding='UTF-8') as fs1:
                for lin in fs1:
                    x.append(lin)
            for i, t in enumerate(c):
                if t == False:
                    fs2.write(x[i])
        fs2.close()
        fs1.close()
        return acc, loss_total / len(data_iter), report, confusion,pre,rec,f1,auc
    return acc, loss_total / len(data_iter),auc,pre,rec,f1
