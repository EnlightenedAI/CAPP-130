# coding: UTF-8
import json
import numpy as np
import torch
import torch.nn as nn
from sklearn import metrics
import time
from utils import get_time_dif
from pytorch_pretrained.optimization import BertAdam
from sklearn.metrics import recall_score,f1_score,precision_score
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
    if config.cintinue:
        model.load_state_dict(torch.load(config.load_path_1))
        model.eval()
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
            loss_func = nn.BCEWithLogitsLoss()
            loss = loss_func(outputs.to(torch.float32), labels.to(torch.float32))
            loss.backward()
            optimizer.step()
            if total_batch % 500 == 0:
                true = labels.data.cpu()
                m=nn.Sigmoid()
                predic = m(outputs.data)
                Threshold=0.5
                predic[predic >Threshold ] = 1
                predic[predic <=Threshold ] = 0
                predic=predic.clone().detach().cpu()
                train_acc = metrics.accuracy_score(true,predic)
                train_pre = precision_score(np.array(true), np.array(predic),average=None,zero_division=0)
                train_rec = recall_score(np.array(true), np.array(predic),average=None,zero_division=0)
                train_f1 = f1_score(np.array(true), np.array(predic),average=None,zero_division=0)
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
    test_acc, test_loss, test_report, test_confusion,pre,rec,f1= evaluate(config, model, test_iter, test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
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
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    loss_func = nn.BCEWithLogitsLoss()
    for texts, labels, lens,masks in data_iter:
        torch.cuda.empty_cache()
        texts=(texts.to(config.device),lens.to(config.device),masks.to(config.device))
        with torch.no_grad():
            outputs = model(texts)
        loss_func=loss_func.to(torch.float32)
        labels=labels.to(torch.float32)
        loss = loss_func(outputs.to('cuda:0'), labels.to('cuda:0'))
        loss_total += loss
        m=nn.Sigmoid()
        predic = m(outputs.data)
        Threshold = 0.5
        predic[predic > Threshold] = 1
        predic[predic <= Threshold] = 0
        if len(labels_all) == 0:
            labels_all=labels.cpu().numpy()
            predict_all=predic.cpu().numpy()
        else:
            labels=labels.cpu().numpy()
            predic = predic.cpu().numpy()
            labels_all = np.vstack((labels_all, labels))
            predict_all = np.vstack((predict_all, predic))
    acc = metrics.accuracy_score(np.array(labels_all.data), np.array(predict_all))
    pre = precision_score(np.array(labels_all), np.array(predict_all),average=None)
    rec = recall_score(np.array(labels_all), np.array(predict_all),average=None)
    f1 = f1_score(np.array(labels_all), np.array(predict_all),average=None)
    if test:
        pre=precision_score(np.array(labels_all), np.array(predict_all),average=None)
        rec=recall_score(np.array(labels_all), np.array(predict_all),average=None)
        f1=f1_score(np.array(labels_all), np.array(predict_all),average=None)
        report = metrics.classification_report(labels_all,predict_all,target_names=['数据采集','权限获取', '共享披露','使用','存储方式','安全措施','特殊人群','权限管理','联系方式','变更','停止运行'])
        print(labels_all.shape,predict_all.shape)
        confusion = metrics.multilabel_confusion_matrix(np.array(labels_all.data), np.array(predict_all))
        return acc, loss_total / len(data_iter), report, confusion,pre,rec,f1
    return acc, loss_total / len(data_iter),pre,rec,f1
