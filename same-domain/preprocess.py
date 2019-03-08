# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import codecs
import os
import re

PKU_TRAIN='original_data/pku_training.utf8'
PKU_TEST='original_data/pku_test_gold.utf8'
MSR_TRAIN='original_data/msr_training.utf8'
MSR_TEST='original_data/msr_test_gold.utf8'
AS_TRAIN='original_data/as_training.utf8'
AS_TEST='original_data/as_test_gold.utf8'
CITYU_TRAIN='original_data/cityu_training.utf8'
CITYU_TEST='original_data/cityu_test_gold.utf8'
CTB_TRAIN='original_data/ctb_train'
CTB_DEV='original_data/ctb_dev'
CTB_TEST='original_data/ctb_test'
CHINESE_IDIOMS='original_data/idioms'
OUTPUT_PATH='data'

rNUM = '(-|\+)?\d+((\.|·)\d+)?%?'
rENG = '[A-Za-z_.]+'
START='S'
END='E'

def strQ2B(ustring):
    """全角转半角"""
    rstring = ""
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 12288:  # 全角空格直接转换
            inside_code = 32
        elif (inside_code >= 65281 and inside_code <= 65374):  # 全角字符（除空格）根据关系转化
            inside_code -= 65248
        rstring += unichr(inside_code)
    return rstring


def preprocess(input,output):
    ''' pku_training.utf8(input) 生成一个pku_train_all(output)文件 '''
    output_filename = os.path.join(OUTPUT_PATH,output)
    idioms=dict()
    #取出成语词典中的成语
    with codecs.open(CHINESE_IDIOMS,'r','utf-8') as f:    
        for line in f:  
            idioms[line.strip()]=1    #返回移除字符串头尾指定的字符(空格)生成的新字符串                              标记为1
    count_idioms = 0
    sents=[]
    #pku_training.utf8的文本处理：
    with codecs.open(input,'r','utf-8') as fin:
        with codecs.open(output_filename,'w','utf-8') as fout:
            for line in fin:   #取pku_training 的每一行
                sent=strQ2B(line).split( ) #此行 词 的集合
                new_sent=[]
                for word in sent:
                    word=re.sub(rNUM,'0',word)  #正则 没懂 ？？？？？？？？？？？？？？？？
                    word=re.sub(rENG,'X',word)
                    if idioms.get(word) is not None:
                        count_idioms+=1
                        word=u'I'
                    new_sent.append(word)
                sents.append(new_sent)
            for sent in sents:
                fout.write('  '.join(sent))
                fout.write('\n')
    #print 'idioms count:%d' % count_idioms

def split(dataset): #划分出验证集
#dataset='pku'
    dataset=os.path.join(OUTPUT_PATH,dataset)
    with codecs.open(dataset+'_train_all','r','utf-8') as f:
	    lines = f.readlines()
	    idx = int(len(lines)*0.9)
	    with codecs.open(dataset+'_train','wb','utf-8') as fo:
		    for line in lines[:idx]:
			    fo.write(line.strip()+'\r')
	    with codecs.open(dataset+'_dev','wb','utf-8') as fo:
		    for line in lines[idx:]:
			    fo.write(line.strip()+'\r')
    os.remove(dataset+'_train_all')


def ngram(ustr,n=2):
    ngram_list=[]
    for i in range(len(ustr)-n+1):
        ngram_list.append(ustr[i:i+n])
    return ngram_list

#处理后的训练集pku_train的bigram大全，包括在训练集中不成词的
def bigram_words(dataset,window_size=2):
#dataset=‘pku_train’
    dataset=os.path.join(OUTPUT_PATH,dataset)
    words=dict()   #空的
    start=''.join([START]*window_size)   # start=‘SS’
    end=''.join([END]*window_size)      #end=‘EE’
    with codecs.open(dataset,'r','utf-8') as f:
        for line in f:   # line="北京  举行  新年  音乐会"
            line=start+re.sub('\s+','',line.strip())+end  #SS北京举行新年音乐会EE
            for word in ngram(line,window_size):
                words[word]=words.get(word,0)+1
            #words={'会E': 1, '北京': 1, '音乐': 1, '乐会': 1, 'EE': 1, '行新': 1, '年音': 1, 'S北': 1, '京举': 1, '举行': 1, '新年': 1, 'SS': 1}    
    with codecs.open(dataset+'_bigram','w','utf-8') as f:
        for k,v in words.items():
            f.write(k+' '+unicode(v)+'\n')


if __name__ == '__main__':
    if not os.path.exists(OUTPUT_PATH):
        os.mkdir(OUTPUT_PATH)
    #preprocess
    #pku_training.utf8 进行处理后生成一个pku_train_all文件
    print('start preprocess')
    preprocess(PKU_TRAIN,'pku_train_all') 
    preprocess(PKU_TEST,'pku_test')
    preprocess(MSR_TRAIN,'msr_train_all')
    preprocess(MSR_TEST,'msr_test')
    preprocess(AS_TRAIN,'as_train_all')
    preprocess(AS_TEST,'as_test')
    preprocess(CITYU_TRAIN,'cityu_train_all')
    preprocess(CITYU_TEST,'cityu_test')
    # preprocess(CTB_TRAIN,'ctb_train')
    # preprocess(CTB_DEV,'ctb_dev')
    # preprocess(CTB_TEST,'ctb_test')

    
    #split
    #划分出验证集,pku_train_all变为pku_train和pku_dev
    print('start split')
    split('pku')  
    split('msr')
    split('as')
    split('cityu')
    
    #bigram
    print('start bigram')
    bigram_words('pku_train')
    bigram_words('msr_train')
    bigram_words('as_train')
    bigram_words('cityu_train')
    # bigram_words('ctb_train')




 
    
