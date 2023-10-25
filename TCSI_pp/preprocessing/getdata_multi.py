import os
import json
import tqdm
from sklearn.model_selection import StratifiedShuffleSplit,train_test_split
from skmultilearn.model_selection.iterative_stratification import iterative_train_test_split
from pprint import pprint
import random
from Test import test
import numpy as np
dataset='all_data'
path= '../'+dataset+'/doccano/'
labelDataFile = '../'+dataset+'/label/label_config.json'
labels_text=open(labelDataFile, 'r', encoding='utf_8').read()
labels_list = json.loads(labels_text)

class_label={}
#获取标签键值对
labelslists=[]
for i, label in enumerate(labels_list):
    class_label.update({label['text']:i+1})
    labelslists.append(label['text'])
labels_list_no_sum=list(set(labelslists).intersection(set(["摘要"])))
datas_url='../'+dataset+'/data/class.txt'
if not os.path.exists(datas_url): 
    os.mkdir(datas_url)
with open(datas_url, 'w', encoding='utf_8') as f_class:
    f_class.write('none'+"\n")
    for labels in labels_list:
        f_class.write(labels['text']+"\n")
    f_class.close()
print(class_label)

testrate=0.4
def merge_json(path_results, path_merges):
    sentences=[]
    ll=[]
    ll_1_11=[]
    f_errors=open('errorsdds.json','w',encoding='utf-8')
    merges_file = os.path.join(path_merges, "bas_fund_transaction.json")
    with open(merges_file, "w", encoding="utf-8") as f0:
        for i,file in enumerate(os.listdir(path_results)):
            if i <130:
                print(i,file)

                with open(os.path.join(path_results, file), "r", encoding="utf-8") as f1:
                    for line in tqdm.tqdm(f1):
                        # print(line)
                        line_dict = json.loads(line)
                        l = [0] * (len(class_label) + 1)
                        if list(set( line_dict["label"]).intersection(set(["摘要"])))==[]:
                            f_errors.write(json.dumps(json.dumps({'id':line_dict["id"],"text":line_dict['text'],'label':line_dict["label"]},ensure_ascii=False)+'\n'))
                            l[0] = 1
                        elif line_dict["label"]==[]:
                            l[0] = 1
                        else:
                            for lab in line_dict["label"]:
                                l[class_label[lab]]=1
                        ll.append(l)
                        sens= ",".join(line_dict["text"].split())
                        sentences.append([sens, l[1:12]])
                        js = json.dumps(line_dict, ensure_ascii=False)
                        f0.write(js + '\n')
            f1.close()
        f0.close()
    return sentences
if __name__ == '__main__':
    path_results, path_merges = '../'+dataset+'/doccano/','../'+dataset+'/results'
    if not os.path.exists(path_merges): 
        os.mkdir(path_merges)
    sentences = merge_json(path_results, path_merges)

    X = np.array([], dtype=np.str_)
    y = np.array([])


    for i,lens in enumerate(sentences):
        if y.size==0:
            if sum(lens[1])!= 0 :
                X=lens[0]
                y=np.array(lens[1])
        else:
            if sum(lens[1]) !=0:
                X=np.append(X,lens[0])
                y=np.vstack((y,np.array(lens[1])))

    np.random.seed(42)
    X_train,y_train, X_test_dev,  y_test_dev =iterative_train_test_split(X, y, test_size=0.3)

    np.random.seed(42)
    X_test, y_test, X_dev, y_dev =iterative_train_test_split(X_test_dev, y_test_dev, test_size=0.5)

    data_url='../'+dataset+'/data/class_multi_iterative'
    if not os.path.exists(data_url):
        os.mkdir(data_url)

    with open(data_url+'/train.txt', "w", encoding="utf-8") as train_text:
        for j in range(X_train.shape[0]):
            train_text.write(str(X_train[j]) + '\t' + str(y_train[j].tolist()) + "\n")
    train_text.close()

    with open(data_url+ '/dev.txt', "w", encoding="utf-8") as dev_text:
        for j in range(X_dev.shape[0]):
            dev_text.write(str(X_dev[j]) + '\t' + str(y_dev[j].tolist()) + "\n")
    dev_text.close()

    with open(data_url+ '/test.txt', "w", encoding="utf-8") as test_text:
        for j in range(X_test.shape[0]):
            test_text.write(str(X_test[j]) + '\t' + str(y_test[j].tolist()) + "\n")
    test_text.close()
