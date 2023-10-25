import json
import jieba
from rouge import Rouge
def rouge_scorces(datafile_url):
    rouge = Rouge()
    datas = open(datafile_url, 'r', encoding='utf_8').readlines()
    datas_all=[]
    for line in datas:
        data=json.loads(line)
        datas_all.append(data)
    hyps , refs = map(list, zip(*[['  '.join(jieba.cut(d['pred'])) if len(d['pred'] ) !=0 else 'is null' , '  '.join(jieba.cut(d['rewrite'])) if len(d['rewrite'] ) !=0 else 'is null'] for d in datas_all]))
    print('_'*10+'rouge'+"_"*10)
    scores = rouge.get_scores(hyps, refs, avg=True)
    return scores

def bert2bert_rouge_scorce(predicted,target):
    rouge = Rouge()
    hyps, refs = map(list, zip(*[['  '.join(jieba.cut(predicted[i])) ,
                                  '  '.join(jieba.cut(target[i]))] for i in
                                 range(len(predicted))]))
    print('_' * 10 + 'mean-rouge' + "_" * 10)
    scores = rouge.get_scores(hyps, refs, avg=True)
    return scores
