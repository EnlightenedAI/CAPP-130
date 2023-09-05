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
if not os.path.exists(datas_url):  # 如果results目录不存在，新建该目录。
    os.mkdir(datas_url)
with open(datas_url, 'w', encoding='utf_8') as f_class:
    f_class.write('none'+"\n")
    for labels in labels_list:
        f_class.write(labels['text']+"\n")
    f_class.close()
# class_label.update({'<PAD>':i+1})
print(class_label)

# files= os.listdir(path)

testrate=0.4
def merge_json(path_results, path_merges):
    """
    主要功能是实现一个目录下的多个json文件合并为一个json文件。
    :param path_results:
    :param path_merges:
    :return:
    """
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
                        # elif list(set( line_dict["label"]).intersection(set(['摘要',"潜在风险"])))==[]:
                        #     None
                        elif line_dict["label"]==[]:
                            l[0] = 1
                        else:
                            for lab in line_dict["label"]:
                                l[class_label[lab]]=1
                        ll.append(l)
                        sens= ",".join(line_dict["text"].split())#句子中的空格改为，
                        sentences.append([sens, l[1:12]])
                        js = json.dumps(line_dict, ensure_ascii=False)
                        f0.write(js + '\n')
            f1.close()
        f0.close()
        # test(sentences,ll)
        # lex = int(len(sentences) * testrate)
        # test1=np.array(ll[0:lex])
        # dev1 = np.array(ll[lex:lex*2])
        # train1 = np.array(ll[lex*2:len(sentences)])
        # print(np.sum(test1,axis=0)/test1.shape[0])
        # print(np.sum(dev1,axis=0)/dev1.shape[0])
        # print(np.sum(train1, axis=0) / train1.shape[0])
    return sentences
if __name__ == '__main__':
    path_results, path_merges = '../'+dataset+'/doccano/','../'+dataset+'/results'
    if not os.path.exists(path_merges):  # 如果results目录不存在，新建该目录。
        os.mkdir(path_merges)
    sentences = merge_json(path_results, path_merges)
    # print(sentences)

    X = np.array([], dtype=np.str_)
    y = np.array([])


    for i,lens in enumerate(sentences):
        # print(y.size)
        if y.size==0:
            if sum(lens[1])!= 0 :
                X=lens[0]
                # print(lens)
                y=np.array(lens[1])
                # print(y)
        else:
            if sum(lens[1]) !=0:
                X=np.append(X,lens[0])
                # print(lens)
                y=np.vstack((y,np.array(lens[1])))
    print(y.shape)
    print(X.shape)
    print(np.sum(y,0))
    # y=np.array(y)
    print("-"*10,'开始切分数据','_'*10)
    print(len(X))

    # 然后，我们可以使用`train_test_split`函数进行分层抽样切分数据集：

    # 切分train和test数据集
    # X_train, X_test_dev, y_train, y_test_dev = train_test_split(X, y, test_size=0.3, random_state=42)
    np.random.seed(42)
    X_train,y_train, X_test_dev,  y_test_dev =iterative_train_test_split(X, y, test_size=0.3)
    # sss = StratifiedShuffleSplit(n_splits=1, test_size=testrate, random_state=0)
    # sss.get_n_splits(X, y)
    # for train_index, test_dev_index in sss.split(X, y):
    #     # print("TRAIN:", train_index, "TEST:", test_index)
    #     X_train, X_test_dev= X[train_index], X[test_dev_index]
    #     y_train, y_test_dev = y[train_index], y[test_dev_index]
    print("-" * 10, '一破！卧龙出山', '_' * 10)
    np.random.seed(42)
    X_test, y_test, X_dev, y_dev =iterative_train_test_split(X_test_dev, y_test_dev, test_size=0.5)
    # X_test, X_dev, y_test, y_dev = train_test_split(X, y, test_size=0.5, random_state=32)
    # sss_T = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=0)
    # sss_T.get_n_splits(X_test_dev, y_test_dev)
    # for dev_index, test_index in sss_T.split(X_test_dev, y_test_dev):
    #     # print("TRAIN:", train_index, "TEST:", test_index)
    #     X_test, X_dev = X_test_dev[dev_index], X_test_dev[test_index]
    #     y_test, y_dev = y_test_dev[dev_index], y_test_dev[test_index]
    print(X_dev.shape)
    print(X_test.shape)
    data_url='../'+dataset+'/data/class_multi_iterative'
    if not os.path.exists(data_url):  # 如果results目录不存在，新建该目录。
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


        #         l_list = map(lambda x: str(x), sen[1])
        #         l_str = ''.join(l_list)
        #         f.write(sen[0] + '\t' + l_str + "\n")
        #     f.close()


        # print(X_train,y_train)
        # print(X_test, y_test)
    # random.shuffle(sentences)
    # # print(sentences)
    # lex=int(len(sentences)*testrate)
    # testdata=sentences[0:lex]
    # devdata=sentences[lex:2*lex]
    # traindata=sentences[2*lex:len(sentences)]
    # with open('./data/test.txt', 'w', encoding='utf_8') as f:
    #     for sen in testdata:
    #         l_list = map(lambda x: str(x), sen[1])
    #         l_str = ''.join(l_list)
    #         f.write(sen[0]+'\t'+l_str+"\n")
    #     f.close()
    # with open('./data/dev.txt', 'w', encoding='utf_8') as f:
    #     for sen in devdata:
    #         l_list = map(lambda x: str(x), sen[1])
    #         l_str = ''.join(l_list)
    #         f.write(sen[0] + '\t' + l_str + "\n")
    #     f.close()
    # with open('./data/train.txt', 'w', encoding='utf_8') as f:
    #     for sen in traindata:
    #         l_list = map(lambda x: str(x), sen[1])
    #         l_str = ''.join(l_list)
    #         f.write(sen[0] + '\t' + l_str + "\n")
    #     f.close()



    # Datas = open(file, 'r', encoding='utf_8').readlines()
    # with open("sentence/"+files[i], encoding='utf-8', mode = "a") as f:
    #     for i in data:
    #         f.write(i + '\n')


