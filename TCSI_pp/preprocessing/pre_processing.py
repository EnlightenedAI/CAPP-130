import os
load_path="load_path/"
save_path="save_path/"
files= os.listdir(path)

def cut_sent(para):
    para = re.sub('([。！？；;\?])([^”’])', r"\1\n\2", para)  # 单字符断句符
    para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
    para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
    para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
    para = para.rstrip()  # 段尾如果有多余的\n就去掉它
    return para.split("\n")
import re
for i in range(len(files)):
    file=load_path+files[i]
    with open(file, encoding='utf-8',  mode = 'r') as lines: 
        array=lines.read()
    data=cut_sent(array)
    with open(save_path+files[i], encoding='utf-8', mode = "w") as f:
        for i in data:
            f.write(i + '\n')