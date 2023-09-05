import json
import random
def split_list_to_nlist(list, sub_list_size):
    num = 0
    tmp = []
    nlist = []
    for i in range(len(list)):
        if num == sub_list_size:
            nlist.append(tmp)
            tmp = []
            num = 0
        tmp.append(list[i])
        num += 1
    if tmp != []:
        nlist.append(tmp)
    return nlist

def dataloading(dataset_url):
    train_datas = open(dataset_url+'train.json', 'r', encoding='utf_8').readlines()
    test_datas = open(dataset_url+'test.json', 'r', encoding='utf_8').readlines()
    dev_datas = open(dataset_url+'dev.json', 'r', encoding='utf_8').readlines()
    train_data=[]
    for line in train_datas:
        data=json.loads(line)
        train_data.append(data)
    test_data = []
    for line in test_datas:
        data = json.loads(line)
        test_data.append(data)
    dev_data = []
    for line in dev_datas:
        data = json.loads(line)
        dev_data.append(data)
    return train_data,dev_data,test_data
def dataloading_only_test(dataset_url):
    test_datas = open(dataset_url, 'r', encoding='utf_8').readlines()
    test_data = []
    for line in test_datas:
        data = json.loads(line)
        test_data.append(data)
    return test_data

def data_write(datas):
    doc_dict = []
    for text_data in datas:
        doc_dict.append((text_data['sentence'], text_data['rewrite']))
    return doc_dict

def dataload(dataset,only_test=False):
    if only_test:
        test_data = dataloading_only_test(dataset)
    else:
        train_data,dev_data,test_data=dataloading(dataset)
    train_datas=data_write(train_data,'train')
    dev_datas=data_write(dev_data,'dev')
    test_datas=data_write(test_data,'test')
    return train_datas,dev_datas,test_datas

