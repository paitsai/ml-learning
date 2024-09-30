# 这是word2vec中文词库的实现方法

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import thulac

def get_word_from_dir(dir_path):
    # 遍历所有的txt文件来训练
    lines=[]
    for filen in os.listdir(dir_path):
        if filen.endswith('.txt'):
            with open(os.path.join(dir_path, filen), 'r', encoding='gbk') as f:
                lines.extend(f.readlines())
    
    data=[]
    for line in lines:
        line=line.strip()
        if(line!=''):
            data.append(line)
            
    with open('input.txt','w',encoding='gbk') as f:
        f.write('\n'.join(data))
        
    #调用thulac库 来将文本切分 不包含词性信息
    thu1 = thulac.thulac(seg_only=True)
    thu1.cut_f("input.txt", "output.txt")
    
    stopwords = '''~!@#$%^&*()_+`1234567890-={}[]:：";'<>,.?/|\、·！（）￥“”‘’《》，。？/—-【】….'''
    stopwords_set = set([i for i in stopwords])
    stopwords_set.add("br") # 异常词也加入此集，方便去除
    
    
    with open("output.txt",'r',encoding='gbk') as f:
        lines=f.readlines()
        
    clean_data=[]
    
    for line in lines:
        line=line.strip()
        for s in stopwords_set:
            line=line.replace(s,"")
            line=line.replace("   "," ").replace("  "," ")
            print(line)
            if  line!="" and line!=" ":
                clean_data.append(line)
                
                
    with open("words.txt","w",encoding="gbk") as f:
        f.write(" ".join(data))
        
                
        

if __name__=="__main__":
    
    get_word_from_dir("./sources_CN") # 请替换为您的txt文本路径
    
            
      

