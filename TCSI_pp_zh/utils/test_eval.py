# coding: UTF-8
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
# from test_utils import get_time_dif
from pytorch_pretrained.optimization import BertAdam
from sklearn.metrics import recall_score,f1_score,precision_score,roc_auc_score
# from loss.focalloss import BCEFocalLoss
# from loss.Focal_loss import focal_loss
import os
# from sklearn.metrics import precision_recall_curve
from numpy.core.fromnumeric import argmax
# import torch.nn as nn

# 权重初始化，默认xavier
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
    # classID=12
    start_time = time.time()
    # model.load_state_dict(torch.load(config.load_path_1))
    # model.eval()

    # checkpoint = torch.load(config.save_path)
    # model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # epoch = checkpoint['epoch']
    # loss = checkpoint['loss']
    # model.train()
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    # optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=config.learning_rate,
                         warmup=0.05,
                         t_total=len(train_iter) * config.num_epochs)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    total_batch: int = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    # model.train()
    # for epoch in range(config.num_epochs):
    #     print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
    #     for i, (trains_1, labels,lens,masks) in enumerate(train_iter):
    #         # print("train",trains_1)
    #         trains=(trains_1.to(config.device),lens.to(config.device),masks.to(config.device))
    #         # print(trains)
    #         outputs = model(trains)
    #         model.zero_grad()
    #         # print(outputs)
    #         # print(labels)
    #         # labels.format("%06d", 25)
    #         # loss = F.cross_entropy(outputs, labels)
    #         # loss =loss1(outputs, labels)
    #         # ----------
    #         # print(labels.shape)
    #         # labels=labels[:,config.num_classes-2:config.num_classes-1]####注意
    #         labels=labels.to(config.device)
    #
    #         # print(labels.shape)
    #         # wight =labels#1
    #         # c = sum(wight)/wight.shape[0]#1
    #         # print("c",c)
    #         # wight[wight == 1] = c.long()#1
    #         # wight[wight == 0] = (1 - c).long()#1            # print(wight)
    #         # print(outputs)
    #         # print(labels.shape)
    #         # t = ((labels.shape[0] - labels.sum(0)) / labels.shape[0])
    #         # weight = torch.zeros_like(labels)
    #         # print(weight)
    #         # print(t[0])
    #         # weight = torch.fill_(weight,  torch.tensor(1-t[0]))
    #         # print(weight)
    #         # weight[labels > 0] = torch.tensor(t[0])
    #         # print(weight)
    #         # criterion_weighted = nn.BCELoss(weight=weight, size_average=True)
    #         # criterion_weighted = nn.BCELoss(weight=wight)#1
    #         # criterion_weighted =BCEFocalLoss()
    #
    #         # criterion_weighted = nn.BCEWithLogitsLoss(weight=torch.tensor([]))
    #         # criterion_weighted =nn.MultiLabelSoftMarginLoss(reduction='mean')
    #         # criterion_weighted=nn.BCELoss(weight=torch.tensor([t,1-t]))
    #         # criterion_weighted = nn.BCEWithLogitsLoss(pos_weight=(y==0.).sum()/y.sum())
    #         # print(outputs)
    #
    #         # criterion_weighted=focal_loss(alpha=config.focalloss_rate, gamma=2, num_classes=2)
    #         # loss = criterion_weighted(outputs, labels)
    #         # print(loss)
    #         # loss = criterion_weighted(outputs.to(torch.float32), labels.to(torch.float32))
    #         # loss1=nn.MultiLabelSoftMarginLoss(weight=None)
    #         # ----------
    #         # loss.backward()
    #         # optimizer.step()
    #         if total_batch % 500 == 0:
    #             # 每多少轮输出在训练集和验证集上的效果
    #             true = labels.data.cpu()
    #             # predic = torch.max(outputs.data, 1)[1].cpu()#原始
    #             # -------
    #             # print("outputs.data",outputs.data)
    #             predic = outputs.data.argmax(axis=1)
    #             predict_true=outputs.data[:,1]
    #             # print("predic",predic)
    #             # print("labels", labels)
    #             # print("predic.shape",predic.shape)
    #             # print("true.shape", true.shape)
    #             # -------
    #             # precision, recall, thresholds = precision_recall_curve(np.array(true), np.array(predic))
    #             # #
    #             # # print(precision)
    #             # # print(recall)
    #             # # print(thresholds)
    #             # target = precision + recall
    #             # index = argmax(target)
    #             # print("p:",precision[index], "\nr:",recall[index],"\nt:",thresholds[index])
    #             # level=0.3
    #             # predic[predic >= level] = 1
    #             # predic[predic <level] = 0
    #
    #             # print(predic)
    #             train_acc = metrics.accuracy_score(true.clone().detach().cpu(), predic.clone().detach().cpu())
    #             train_pre = precision_score(np.array(true.clone().detach().cpu()), np.array(predic.clone().detach().cpu()),average=None)
    #             train_rec = recall_score(np.array(true.clone().detach().cpu()), np.array(predic.clone().detach().cpu()),average=None)
    #             train_f1 = f1_score(np.array(true.clone().detach().cpu()), np.array(predic.clone().detach().cpu()),average=None)
    #             # print("train_acc",train_acc)
    #             # train_auc=roc_auc_score(true.clone().detach().cpu(), predict_true.clone().detach().cpu())
    #             train_auc ="none"
    #             dev_acc, dev_loss,dev_auc,dev_pre,dev_rec,dev_f1= evaluate(config, model, dev_iter)
    #             if dev_loss < dev_best_loss:
    #                 dev_best_loss = dev_loss
    #
    #                 torch.save(model.state_dict(), config.save_path)
    #                 improve = '*'
    #                 last_improve = total_batch
    #             else:
    #                 improve = ''
    #             time_dif = get_time_dif(start_time)
    #             msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
    #             tra_msg='Tra_Pre: {0:>6.2%},    Tra_Rec: {1:>6.2%},    Tra_F1: {2:>6.2%},   Val_Pre: {3:>6.2%},    Val_Rec: {4:>6.2%},   Val_F1: {5:>6.2%}'
    #             with open(config.save_path_acc_loss,"a",encoding="utf-8") as write_loss:
    #                 write_loss.write(json.dumps({'Iter': str(total_batch),'Train Loss': str(loss.item()),'Train Acc': str(train_acc),
    #                                              'Val Loss': str(dev_loss.item()),'Val Acc': str(dev_acc),'Time': str(time_dif),
    #                                              'train_Precision': str(train_pre),'train_Recall': str(train_rec),'train_F1': str(train_f1),
    #                                              'Val_Precision': str(dev_pre),'Val_Recall': str(dev_rec), 'Val_F1': str(dev_f1)},ensure_ascii=False)+ '\n')
    #
    #             write_loss.close()
    #             print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
    #             # print(tra_msg.format(str(train_pre),str(train_rec),str(train_f1),str(dev_pre), str(dev_rec), str(dev_f1)))
    #
    #
    #             # with open("./CLASS_Data/output/trainput/"+config.dataset+"_"+config.model_name+"_focalloss_rate_"+str(config.focalloss_rate).replace(str(config.focalloss_rate)[1], "", 1)+".json", "a") as train_out:
    #             #     train_out.write(json.dumps({'train_acc':train_acc,'train_acc':train_auc,'dev_acc':dev_acc,'dev_auc':dev_auc})+'\n')
    #             model.train()
    #         total_batch += 1
    #         if total_batch - last_improve > config.require_improvement:
    #             # 验证集loss超过1000batch没下降，结束训练
    #             print("No optimization for a long time, auto-stopping...")
    #             flag = True
    #             break
    #     if flag:
    #         break
    test(config, model, test_iter)


def test(config, model, test_iter):
    # test
    return evaluate(config, model, test_iter, test=True)





def evaluate(config, model, data_iter, test=False):
    # classID=12
    model.eval()
    loss_total = 0
    predict_all=np.empty(shape=[1, 0], dtype=int)
    predict_id=np.empty(shape=[1, 0], dtype=int)
    # labels_all=np.empty(shape=[1, len(config.class_list)], dtype=int)
    # predict_all=[]
    # labels_all=[]
    with torch.no_grad():
        for texts, lens,masks ,id in data_iter:
            texts=(texts.to(config.device),lens.to(config.device),masks.to(config.device))
            outputs = model(texts)

            predic = outputs.data.argmax(axis=1)
            predict_all = np.append(predict_all, predic.cpu().numpy())
            predict_id =np.append( predict_id, id.cpu().numpy())
    return predict_all,predict_id



