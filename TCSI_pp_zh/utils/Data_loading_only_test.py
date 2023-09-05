import json
import random
# dataset='./out_data.json'
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
    datas = open(dataset_url, 'r', encoding='utf_8').readlines()
    datas_all=[]
    for line in datas:
        data=json.loads(line)
        datas_all.append(data)
    test_data=datas_all
    return test_data

def data_write(datas):
    doc_dict = []
    print(len(datas))
    for text_data in datas:
        doc_dict.append((text_data['text'],text_data['label'],text_data['highlight']))
    return doc_dict



def dataload(dataset):
    test_data=dataloading(dataset)
    test_datas=data_write(test_data)
    print('''3''')
    return test_datas

