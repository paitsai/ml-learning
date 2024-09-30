import os

from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import string
from autocorrect import Speller
import re
from nltk.corpus import stopwords


def restore_common_abbr(caption):
    # 还原常见缩写单词
    pat_is = re.compile("(it|he|she|that|this|there|here)(\'s)", re.I)
    pat_s = re.compile("(?<=[a-zA-Z])\'s")  # 找出字母后面的字母
    pat_s2 = re.compile("(?<=s)\'s?")
    pat_not = re.compile("(?<=[a-zA-Z])n\'t")  # not的缩写
    pat_would = re.compile("(?<=[a-zA-Z])\'d")  # would的缩写
    pat_will = re.compile("(?<=[a-zA-Z])\'ll")  # will的缩写
    pat_am = re.compile("(?<=[I|i])\'m")  # am的缩写
    pat_are = re.compile("(?<=[a-zA-Z])\'re")  # are的缩写
    pat_ve = re.compile("(?<=[a-zA-Z])\'ve")  # have的缩写

    new_text = caption
    new_text = pat_is.sub(r"\1 is", new_text)
    new_text = pat_s.sub("", new_text)
    new_text = pat_s2.sub("", new_text)
    new_text = pat_not.sub(" not", new_text)
    new_text = pat_would.sub(" would", new_text)
    new_text = pat_will.sub(" will", new_text)
    new_text = pat_am.sub(" am", new_text)
    new_text = pat_are.sub(" are", new_text)
    new_text = pat_ve.sub(" have", new_text)
    new_text = new_text.replace('\'', ' ')
    return new_text


def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None
		
def prepare(caption,stopword,lemmatizer):
    caption = caption.replace('/', ' or ').replace('&', ' and ')
    caption = " ".join(caption.split()).lower()  # replace multiple spaces
    caption = restore_common_abbr(caption)

    doc = word_tokenize(caption)
    corrector=Speller()
    doc = [corrector.autocorrect_sentence(c) if c not in string.punctuation else c for c in doc]
    if doc[-1] != '.':
        doc.append('.')
    tagged_sent = pos_tag(doc)  # 获取单词词性

    new_s = []
    for c in tagged_sent:
        if c[0].isdigit():
            new_s.append("#number")
        elif c[0] not in string.punctuation:
            if c[0] not in stopword:
                wordnet_pos = get_wordnet_pos(c[1]) or wordnet.NOUN #确定词性 如果无法确认则默认为名词性质
                new_s.append(lemmatizer.lemmatize(c[0], pos=wordnet_pos))  # 词形还原

    return " ".join(new_s)



def get_word_from_dir(dir_path):
    # 遍历所有的txt文件来训练
    stopword = stopwords.words('english')
    
    lemmatizer = WordNetLemmatizer()# 进行词形还原
    lines=[]
    for filen in os.listdir(dir_path):
        if filen.endswith('.txt'):
            with open(os.path.join(dir_path, filen), 'r', encoding='utf-8',errors='ignore') as f:
                lines.extend(f.readlines())
    clean_words=[]
    for line in lines:
        line=line.strip()#掐头去尾
        if line=='':
            pass
        else:
            caption = prepare(line,stopword=stopword,lemmatizer=lemmatizer)
            clean_words.append(caption)#得需要二维的list更好
            print(caption)
    
    with open("words_EN.txt",'w',encoding='utf-8') as f:
        f.write('\n'.join(clean_words))
            
        
    




if __name__ == '__main__':
    stopword = stopwords.words('english')
    
    lemmatizer = WordNetLemmatizer()# 进行词形还原
    
    get_word_from_dir("./sources_EN")