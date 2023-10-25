import os
import json
import tqdm
from sklearn.model_selection import StratifiedShuffleSplit
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from pprint import pprint
import random
from Test import test
import numpy as np
from sklearn.utils import resample
dataset='all_data'
path= '../'+dataset+'/doccano/'
labelDataFile = '../'+dataset+'/label/label_config.json'
labels_text=open(labelDataFile, 'r', encoding='utf_8').read()
labels_list = json.loads(labels_text)
class_label={}

for i, label in enumerate(labels_list):
    class_label.update({label['text']:i+1})
datas_url='../'+dataset+'/data/class.txt'
if not os.path.exists(datas_url):  
    os.mkdir(datas_url)
with open(datas_url, 'w', encoding='utf_8') as f_class:
    f_class.write('none'+"\n")
    for labels in labels_list:
        f_class.write(labels['text']+"\n")
    f_class.close()

testrate=0.15
def merge_json(path_results, path_merges):
    sentences=[]
    ll=[]
    f_errors=open('errorsdds.json','w',encoding='utf-8')
    merges_file = os.path.join(path_merges, "bas_fund_transaction.json")
    with open(merges_file, "w", encoding="utf-8") as f0:
        for i,file in enumerate(os.listdir(path_results)):
            print(i,file)
            if i<150:
                with open(os.path.join(path_results, file), "r", encoding="utf-8") as f1:
                    for line in tqdm.tqdm(f1):
                        line_dict = json.loads(line)
                        l = [0] * (len(class_label) + 1)
                        if list(set( line_dict["label"]).intersection(set(["摘要"])))==[]:
                            f_errors.write(json.dumps(json.dumps({'id':line_dict["id"],"text":line_dict['text'],'label':line_dict["label"]},ensure_ascii=False)+'\n'))
                            l[0] = 1
                        elif line_dict["label"]==[]:
                            l[0]=1
                        else:
                            for lab in line_dict["label"]:
                                l[class_label[lab]]=1
                        ll.append(l)
                        sens= line_dict["text"].replace(' ', '，')
                        sens=json.dumps({"text":sens,"rewrite":line_dict['rewrite']}, ensure_ascii=False)###########
                        sentences.append([sens, l])
                        js = json.dumps(line_dict, ensure_ascii=False)
                        f0.write(js + '\n')
                print(len(sentences))
                f1.close()
    f0.close()
    return sentences
    
if __name__ == '__main__':
    path_results, path_merges = '../'+dataset+'/doccano/','../'+dataset+'/results'
    if not os.path.exists(path_merges): 
        os.mkdir(path_merges)
    sentences = merge_json(path_results, path_merges)
    print(len(sentences))

    for i in range(12,13):
        X = np.array([], dtype=np.str_)
        y = np.array([])
        for lens in sentences:
            if i==0:
                X = np.append(X, lens[0])
                if  lens[1][12]==0:
                    y = np.append(y,0)
                else:
                    y = np.append(y, lens[1][i])
            elif i==12:
                X = np.append(X, lens[0])
                y = np.append(y, lens[1][i])
            else:
                if lens[1][12]==1:
                    X=np.append(X,lens[0])
                    y=np.append(y,lens[1][i])
        sss = StratifiedShuffleSplit(n_splits=1, test_size=testrate*2, random_state=0)
        sss.get_n_splits(X, y)
        
        for train_index, test_dev_index in sss.split(X, y):
            X_train, X_test_dev = X[train_index], X[test_dev_index]
            print(X_train.shape)
            print(X_test_dev.shape)
            y_train, y_test_dev = y[train_index], y[test_dev_index]

        X_train_a = np.array([], dtype=np.str_)
        y_train_a = np.array([])
        X_train_o = np.array([], dtype=np.str_)
        y_train_o = np.array([])
        
        for sentence,y_a in zip(X_train,y_train):
            x_a=json.loads(sentence)
            x_a["text"] = x_a["text"].replace("\t", '，')
            x_a["text"]=x_a["text"].replace(" ", '，')
            X_train_a = np.append(X_train_a,x_a["text"])
            y_train_a = np.append(y_train_a,y_a)
            X_train_o = np.append(X_train_o, x_a["text"])
            y_train_o = np.append(y_train_o, y_a)
            if not x_a["rewrite"]==[]:
                X_train_a=np.append(X_train_a,x_a["rewrite"])
                y_train_a = np.append(y_train_a,y_a)

        data = np.column_stack((X_train_a, y_train_a))
        print(data)
        class_0 = data[data[:, -1] == '0.0']
        class_1 = data[data[:, -1] == '1.0']
       
        # 欠采样多数类别
        if i==12:
            class_0_undersampled = resample(class_1, replace=False, n_samples=len(class_0), random_state=42)
            # 过采样少数类别（class_1）
            class_1_oversampled = resample(class_0, replace=True, n_samples=len(class_1), random_state=42)
        else:
            class_0_undersampled = resample(class_0, replace=False, n_samples=len(class_1), random_state=42)
            # 过采样少数类别（class_1）
            class_1_oversampled = resample(class_1, replace=True, n_samples=len(class_0), random_state=42)
        # 合并采样后的数据
        train_un_data = np.vstack((class_0_undersampled, class_1))

        train_over_data=np.vstack((class_0, class_1_oversampled))

        np.random.seed(42)
        np.random.shuffle(train_over_data)
        np.random.shuffle(train_un_data)

        X_train_over = train_over_data[:, :-1]  
        y_train_over = train_over_data[:, -1]

        X_train_un = train_un_data[:, :-1]  
        y_train_un = train_un_data[:, -1]

        # #过采样
        # oversampler = RandomOverSampler(random_state=42)
        #
        # X_resampled, y_resampled = oversampler.fit_resample(X_train, y_train)
        # # 均衡采样
        # sampler = SMOTEENN(random_state=42)
        # X_sm, y_sm= sampler.fit_resample(X_train, y_train)
        # #欠采样
        # undersampler = RandomUnderSampler(random_state=42)
        # X_un, y_un = undersampler.fit_resample(X_train, y_train)

        sss_T = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=0)
        sss_T.get_n_splits(X_test_dev, y_test_dev)
        for dev_index, test_index in sss_T.split(X_test_dev, y_test_dev):
            X_dev, X_test = X_test_dev[dev_index], X_test_dev[test_index]
            y_dev, y_test = y_test_dev[dev_index], y_test_dev[test_index]
            print(X_dev.shape)
            print(X_test.shape)

        X_dev_a = np.array([], dtype=np.str_)
        y_dev_a = np.array([])
        X_dev_o = np.array([], dtype=np.str_)
        y_dev_o = np.array([])
        for sentence,y_a in zip(X_dev,y_dev):
            x_a=json.loads(sentence)
            x_a["text"] = x_a["text"].replace("\t", '，')
            x_a["text"]=x_a["text"].replace(" ", '，')
            X_dev_a = np.append(X_dev_a,x_a["text"])
            y_dev_a = np.append(y_dev_a,y_a)
            X_dev_o = np.append(X_dev_o, x_a["text"])
            y_dev_o = np.append(y_dev_o, y_a)
            if not x_a["rewrite"]==[]:
                X_dev_a=np.append(X_dev_a,x_a["rewrite"])
                y_dev_a = np.append(y_dev_a,y_a)


        data = np.column_stack((X_dev_a, y_dev_a))
        class_0 = data[data[:, -1] == '0.0']
        class_1 = data[data[:, -1] == '1.0']
        print(i)
        print("class0.shape", class_0.shape)
        print("class1.shape", class_1.shape)
        # 欠采样多数类别
        if i==12:
            class_0_undersampled = resample(class_1, replace=False, n_samples=len(class_0), random_state=42)
            # 过采样少数类别（class_1）
            class_1_oversampled = resample(class_0, replace=True, n_samples=len(class_1), random_state=42)
        else:
            class_0_undersampled = resample(class_0, replace=False, n_samples=len(class_1), random_state=42)
            # 过采样少数类别（class_1）
            class_1_oversampled = resample(class_1, replace=True, n_samples=len(class_0), random_state=42)
        # 合并采样后的数据
        dev_un_data = np.vstack((class_0_undersampled, class_1))
        dev_over_data=np.vstack((class_0, class_1_oversampled))
        np.random.seed(42)
        np.random.shuffle(dev_un_data)
        np.random.shuffle(dev_over_data)
        X_dev_over = dev_over_data[:, :-1] 
        y_dev_over = dev_over_data[:, -1]

        X_dev_un = dev_un_data[:, :-1] 
        y_dev_un = dev_un_data[:, -1]

        X_test_o = np.array([], dtype=np.str_)
        y_test_o = np.array([])
        for sentence, y_a in zip(X_test, y_test):
            x_a = json.loads(sentence)
            x_a["text"] = x_a["text"].replace("\t", '，')
            x_a["text"]=x_a["text"].replace(" ", '，')
            X_test_o = np.append(X_test_o, x_a["text"])
            y_test_o = np.append(y_test_o, y_a)
        data_url='../'+dataset+'/data_a/class'+str(i)
        if not os.path.exists(data_url): 
            os.mkdir(data_url)

        with open(data_url+'/train.txt', "w", encoding="utf-8") as train_text:
            for j in range(X_train_o.shape[0]):
                train_text.write(str(X_train_o[j]) + '\t' + str(int(y_train_o[j])) + "\n")
        train_text.close()

        with open(data_url+ '/dev.txt', "w", encoding="utf-8") as dev_text:
            for j in range(X_dev_o.shape[0]):
                dev_text.write(str(X_dev_o[j]) + '\t' + str(int(y_dev_o[j])) + "\n")
        dev_text.close()

        with open(data_url+ '/test.txt', "w", encoding="utf-8") as test_text:
            for j in range(X_test_o.shape[0]):
                test_text.write(str(X_test_o[j]) + '\t' + str(int(y_test_o[j])) + "\n")
        test_text.close()

        with open(data_url + '/trainover.txt', "w", encoding="utf-8") as trainover_text:
            for j in range(X_train_over.shape[0]):
                trainover_text.write(str(X_train_over[j]) + '\t' + str(int(float(y_train_over[j]))) + "\n")
        trainover_text.close()
        with open(data_url + '/trainun.txt', "w", encoding="utf-8") as trainun_text:
            for j in range(X_train_un.shape[0]):
                trainun_text.write(str(X_train_un[j]) + '\t' + str(int(float(y_train_un[j]))) + "\n")
        trainun_text.close()

        with open(data_url + '/devover.txt', "w", encoding="utf-8") as devover_text:
            for j in range(X_dev_over.shape[0]):
                devover_text.write(str(X_dev_over[j]) + '\t' + str(int(float(y_dev_over[j]))) + "\n")
        devover_text.close()
        with open(data_url + '/devun.txt', "w", encoding="utf-8") as devun_text:
            for j in range(X_dev_un.shape[0]):
                devun_text.write(str(X_dev_un[j]) + '\t' + str(int(float(y_dev_un[j]))) + "\n")
        devun_text.close()
