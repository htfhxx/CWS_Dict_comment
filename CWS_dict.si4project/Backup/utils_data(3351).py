# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from collections import Counter
import gensim
import numpy as np
import codecs
import os
#import cPickle
import re
import preprocess
import random
random.seed(8)

UNK='U'
PAD='P'
START='S'
END='E'
TAGB,TAGI,TAGE,TAGS=0,1,2,3
DATA_PATH='data'

#convert word to tag
#当word="李","北京","哈尔滨"，"呼和浩特"
#各个word对应的word2tag(word): [3] [0, 2] [0, 1, 2] [0, 1, 1, 2]
def word2tag(word):

    if len(word)==1:
        return [TAGS]
    if len(word)==2:
        return [TAGB,TAGE]
    tag=[]
    tag.append(TAGB)
    for i in range(1,len(word)-1):
        tag.append(TAGI)
    tag.append(TAGE)
    return tag

#get words from dictionaries
def get_words(general_words_path,domain_words_path=None):
    word_lists=dict()
    with codecs.open(general_words_path,'r','utf-8') as f:
        for line in f:
            line=line.strip().split()[0]
            word_lists[line]=1
    if domain_words_path is not None:
        with codecs.open(domain_words_path,'r','utf-8') as f:
            for line in f:
                line = line.strip().split()[0]
                word_lists[line] = 1
    return word_lists
    
#赋予各个词id  包括训练集的字、bigram的词
#get dictionary from character and bigram to id  
def get_word2id(filename,bigram_words=None,min_bw_frequence=0):
    #word2id = get_word2id(train_data_path,bigram_words_path,FLAGS.min_bg_freq)
    #                       pku_train   pku_train_bigram  0
    filename=os.path.join(DATA_PATH,filename)   # data/pku_train
    # 训练集的一个个字加到x后
    x=[UNK,PAD,START,END] #[UPSE]
    with codecs.open(filename,'r','utf-8') as f:
        for line in f:
            word_list=line.strip().split()
            for word in word_list:
                x.extend([c for c in word])
    #pku_train_bigram的一个个bigram加入bigrams列表中            
    bigrams=[]
    if bigram_words is not None:
        bigram_words=os.path.join('data',bigram_words)
        with codecs.open(bigram_words,'r','utf-8') as f:
            for line in f:
                com=line.strip().split()
                #filter some low frequency bigram words
                if int(com[1])>min_bw_frequence:
                    bigrams.append(com[0])
    # x包括了 训练集的一个个字   ，以及bigram的一个个词（不包括bigram的数量）  （后者与前者不对应）              
    x.extend(bigrams)
    #由列表变成一个集合，Counter({'E': 2, 'P': 1, 'U': 1, 'S': 1})形式
    counter = Counter(x)
    #counter排序,频率降序，字符升序
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    
    words, _ = list(zip(*count_pairs)) #与 zip 相反，*zipped（*count_pairs）可理解为解压，返回二维矩阵式
    #example:  word_to_id={'P': 1, 'E': 0, 'U': 3, 'S': 2}  赋予各个词id
    word_to_id = dict(zip(words, range(len(words))))
    
    return word_to_id

#get reverse_dictionary from id to character or bigram
def build_reverse_dictionary(word_to_id):
    reverse_dictionary = dict(zip(word_to_id.values(), word_to_id.keys()))
    return reverse_dictionary

#get model's inputs
#例如：每行，line="李  鹏  在  北京  考察  企业"
    
def get_train_data(filename=None,word2id=None,usebigram=True):
    #X_train,y_train=get_train_data(train_data_path,word2id)
    #                                   pku_train        
    filename=os.path.join(DATA_PATH,filename) # data/pku_train
    x,y=[],[]
    with codecs.open(filename,'r','utf-8') as f:
        for line in f:
            word_list=line.strip().split()  #每行单词存到list中
            line_y=[]
            line_x=[]
            '''
            当line="李  鹏  在  北京  考察  企业"
            各个word对应的line_y(word2tag(word)):
            [3]
            [3]
            [3]
            [0, 2]
            [0, 2]
            [0, 2]
            三个or四个词为：[0, 1, 1, 2]、[0, 1, 2]
            '''
            for word in word_list:
                line_y.extend(word2tag(word))
            #line_y:[3,3, 3, 0, 2, 0, 2, 0, 2]
            y.append(line_y)  #多行的line_y都放入y中
            line=re.sub(u'\s+','',line.strip()) #去掉line里的空格：李鹏在北京考察企业
            
            contexs=window(line) #句子line中的所有五字的词，有len(line)个
            #contexs=['SS李鹏在', 'S李鹏在北', '李鹏在北京', '鹏在北京考', '在北京考察', '北京考察企', '京考察企业', '考察企业E', '察企业EE']

            #word2id是函数的参数，代表各个字的id           
            #例子：word2id={'P': 1, 'E': 0, 'U': 3, 'S': 2,'李': 5,'鹏': 6, '在': 7, '北': 8,'京'：4}            
            for contex in contexs:
                charx=[]
                
                #contex window
                #将contex，转换成word2id中的id。contex='李鹏在北京'; charx.extend([5, 6, 7, 8, 4])
                charx.extend([word2id.get(c,word2id[UNK]) for c in contex])

                #bigram feature
                #contex='李鹏在北京',在word2id中找到contex的bigram的id，不能就返回U的id。
                #contex='李鹏在北京'; bigram=['李鹏', '鹏在', '在北', '北京']
                
                
                if usebigram:
                    charx.extend([word2id.get(bigram,word2id[UNK]) for bigram in preprocess.ngram(contex)])
                #最终结果放入line_x ：[5, 6, 7, 8, 4, 3, 3, 3, 3]
                line_x.append(charx)

            #最终的x是n(训练集的字数)个9维的列表组成的list。
            #8维数据中包含：以某个字为中心，窗口大小为5的词,包含的五个字的id，以及这个size=5窗口得到的4个bigram的id
            x.append(line_x)
            assert len(line_x)==len(line_y)
    return x,y


#得到句子ustr中的所有五字的词，有len(ustr)个
#李鹏在北京考察企业
#['SS李鹏在', 'S李鹏在北', '李鹏在北京', '鹏在北京考', '在北京考察', '北京考察企', '京考察企业', '考察企业E', '察企业EE']
def window(ustr,left=2,right=2):
    ''''
    第一步：SS李鹏在北京考察企业李鹏在北京考察企业SS
    第二步：SS李鹏在
            S李鹏在北
            李鹏在北京
            鹏在北京考
            在北京考察
            北京考察企
            京考察企业
            考察企业E
            察企业EE
    '''
    sent=''
    for i in range(left):
        sent+=START
    sent+=ustr
    for i in range(right):
        sent+=END
    windows=[]
    for i in range(len(ustr)):
        windows.append(sent[i:i+left+right+1])
    return windows

def tag_sentence(sentence,words,user_words=None):
    '''
    feature vector 
    '''
    word_list=words
    if user_words is not None:
        for word in user_words:
            word_list[word]=1
    result=[]
    for i in range(len(sentence)):
        #fw
        word_tag=[]
        for j in range(4,0,-1):
            if (i-j)<0:
                word_tag.append(0)
                continue
            word=''.join(sentence[i-j:i+1])
            if word_list.get(word) is not None:
                word_tag.append(1)
            else:
                word_tag.append(0)
        #bw
        for j in range(1,5):
            if (i+j)>=len(sentence):
                word_tag.append(0)
                continue
            word=''.join(sentence[i:i+j+1])
            if word_list.get(word) is not None:
                word_tag.append(1)
            else:
                word_tag.append(0)
        result.append(word_tag)
    return result

def tag_documents(filename,words):
    filename=os.path.join(filename)
    result=[]
    with codecs.open(filename,'r','utf-8') as f:
        for line in f:
            word_list=line.strip().split()
            line_x=[]
            for word in word_list:
                line_x.extend([c for c in word])
            result.append(tag_sentence(line_x,words))
    return result

#get dictionary feature vector
def generate_dicttag(filename,general_words_path='dict_1',domain_words_path=None,p=1.0):
    filename=os.path.join(DATA_PATH,filename)
    general_words_path=os.path.join(DATA_PATH,general_words_path)
    if domain_words_path is None:
        domain_words_path=None
    else:
        domain_words_path=os.path.join(DATA_PATH,domain_words_path)
    word_lists=get_words(general_words_path,domain_words_path)
    all_words = word_lists.keys()
    random.shuffle(all_words)
    words=all_words[:int(len(word_lists) * p)]
    new_word_lists = dict()
    for word in words:
        new_word_lists[word] = 1
    data=tag_documents(filename,new_word_lists)
    return data

#get pre-trained embeddings
def get_embedding(word2id,size=100):
    fname='data/wordvec_'+str(size)
    init_embedding = np.zeros(shape=[len(word2id), size])
    pre_trained=gensim.models.KeyedVectors.load(fname)
    pre_trained_vocab = set([unicode(w.decode('utf8')) for w in pre_trained.vocab.keys()])
    c=0
    for word in word2id.keys():
        if len(word)==1:
            if word in pre_trained_vocab:
                init_embedding[word2id[word]]=pre_trained[word.encode('utf-8')]
            else:
                init_embedding[word2id[word]]=np.random.uniform(-0.5,0.5,size)
                c+=1
    for word in word2id.keys():
        if len(word)==2:
            init_embedding[word2id[word]]=(init_embedding[word2id[word[0]]]+init_embedding[word2id[word[1]]])/2
    init_embedding[word2id[PAD]]=np.zeros(shape=size)
    print('oov character rate %f' % (float(c)/len(word2id)))
    return init_embedding

if __name__ == '__main__':
    train_filename='msr_train'
    test_filename='msr_test'
    word2id=get_word2id(train_filename,'msr_train_bigram',min_bw_frequence=1)
    id2word=dict([(y,x) for (x,y) in word2id.items()])
    dict=get_words('data/dict_1')
    cPickle.dump([word2id,id2word,dict],open('checkpoints/maps_msr.pkl','wb'),cPickle.HIGHEST_PROTOCOL)
    # print len(word2id)
    # init_embedding=get_embedding(word2id,100)
    # print init_embedding.shape
    # x,y=get_train_data(test_filename,word2id)
    # d=generate_dicttag(test_filename,domain_words_path=None)
    # print x[0]
    # print y[0]
    # for i in range(len(x)):
    #     assert len(x[i])==len(y[i])
    # print ''.join([id2word[x[2]] for x in x[0]])

    















