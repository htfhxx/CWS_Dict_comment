# -*- coding: utf-8 -*-
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import time
import itertools
import models
import tensorflow as tf
import numpy as np
from config import *
from utils_data import *
from utils import *
from sklearn.model_selection import train_test_split


tf.flags.DEFINE_string('dataset','msr',"Dataset for evaluation")
tf.flags.DEFINE_string("model_path", 'msr2', "The filename of model path")
tf.flags.DEFINE_float("memory",1.0,"Allowing GPU memory growth")
tf.flags.DEFINE_bool('is_train',True,"Train or predict")
tf.flags.DEFINE_integer('min_bg_freq',0,'The mininum bigram_words frequency')
tf.flags.DEFINE_string('general','dict_1',"General dictionary.")
tf.flags.DEFINE_string('domain',None,"domain-specific dictionary")
tf.flags.DEFINE_string('model','DictHyperModel','Choose the model.')
FLAGS = tf.flags.FLAGS

train_data_path=FLAGS.dataset+'_train'
dev_data_path=FLAGS.dataset+'_dev'
test_data_path=FLAGS.dataset+'_test'
bigram_words_path=FLAGS.dataset+'_train_bigram'
config=DictConfig

if FLAGS.dataset == 'pku':
    config.hidden_dim = 64
if FLAGS.dataset == 'msr' or FLAGS.dataset=='as':
    FLAGS.min_bg_freq = 1
if FLAGS.dataset == 'as' or FLAGS.dataset=='cityu':
    FLAGS.domain = 'dict_2'
    #dict_1 (Simplified Chinese dictionary from jieba)
    #dict_2 (Traditional Chinese dictionary from Taiwan version of jieba ) 

def train():
    #对训练集的各个字和生成的bigram 赋予id  
    #例子：word2id={'P': 1, 'E': 0, 'U': 3, 'S': 2,'李': 5,'鹏': 6, '在': 7, '北': 8,'京'：4}
    word2id = get_word2id(train_data_path,bigram_words=bigram_words_path,min_bw_frequence=FLAGS.min_bg_freq)

    #最终的X_train是n(训练集的字数)个9维的列表组成的list。
    #X_train 9维数据中包含：以某个字为中心，窗口大小为5的词,包含的五个字的id，以及这个size=5窗口得到的4个bigram的id     
    #y_train是各个字对应的分词标记BIES：TAGB,TAGI,TAGE,TAGS=0,1,2,3
    X_train,y_train=get_train_data(train_data_path,word2id)
    X_valid,y_valid=get_train_data(dev_data_path,word2id)
    x_test, y_test = get_train_data(test_data_path, word2id)

    #得到训练集每个字所对应的特征向量：
    #代表以第i字为结尾的五字四字三字二字的词                  / 以第i字为开头的二字三字四字五字的词 
    dict_train=generate_dicttag(train_data_path,general_words_path=FLAGS.general,domain_words_path=FLAGS.domain)
    dict_valid=generate_dicttag(dev_data_path,general_words_path=FLAGS.general,domain_words_path=FLAGS.domain)
    dict_test=generate_dicttag(test_data_path,general_words_path=FLAGS.general,domain_words_path=FLAGS.domain)

    #词向量矩阵 len(word2id * size)
    init_embedding = get_embedding(word2id,size=config.word_dim)

    print( 'train_data_path: %s' % train_data_path )
    print( 'valid_data_path: %s' % dev_data_path)
    print( 'test_data_path: %s' % test_data_path)
    print( 'bigram_words_path: %s' % bigram_words_path)
    print( 'model_path: %s' % FLAGS.model_path)
    print( 'min_bg_freq: %d'% FLAGS.min_bg_freq)

    print( 'len(train_data): %d' % len(X_train))
    print( 'len(valid_data): %d' % len(X_valid))
    print( 'len(test_data): %d' % len(x_test))
    print( 'init_embedding shape: [%d,%d]' % (init_embedding.shape[0], init_embedding.shape[1]))
    print( 'Train started!')
    print(FLAGS.is_train)
	
    
    tfConfig = tf.ConfigProto()
    tfConfig.gpu_options.per_process_gpu_memory_fraction = FLAGS.memory
    with tf.Session(config=tfConfig) as sess:
        model=getattr(models,FLAGS.model)(vocab_size=len(word2id),word_dim=config.word_dim,hidden_dim=config.hidden_dim,
                    pad_word=word2id[PAD],init_embedding=init_embedding,num_classes=config.num_classes,clip=config.clip,
                    lr=config.lr,l2_reg_lamda=config.l2_reg_lamda,num_layers=config.num_layers,rnn_cell=config.rnn_cell,
                    bi_direction=config.bi_direction,hidden_dim2=config.hidden_dim2,hyper_embedding_size=config.hyper_embed_size)


		#保存断点 checkpoints
        if not os.path.exists('checkpoints'):
            os.mkdir('checkpoints')
        checkpoints_model=os.path.join('checkpoints',FLAGS.model_path)  #checkpoints_model断点路径
        saver = tf.train.Saver(tf.global_variables ())		
        ckpt = tf.train.get_checkpoint_state(checkpoints_model) #如果“checkpoint”文件包含有效的CheckpointState原型，则返回它
       
	    #接着训练or从头开始训练
        if ckpt and ckpt.model_checkpoint_path:
            print( 'restore from original model!')
            saver.restore(sess, ckpt.model_checkpoint_path)    #恢复变量
        else:
            sess.run(tf.global_variables_initializer())
		
		
        best_f1,best_e=0,0
        for epoch in range(config.n_epoch):
            start_time=time.time()  
            #
            #train
            train_loss=[]
            for step,(X,dict_X,Y) in enumerate(data_iterator2(zip(X_train,dict_train),y_train,128,padding_word=word2id[PAD],shuffle=True)):
                
				#训练得到loss
                loss=model.train_step(sess,X,dict_X,Y,config.dropout_keep_prob)
				
                print( 'epoch:%d>>%2.2f%%' % (epoch,config.batch_size*step*100.0/len(X_train)),'completed in %.2f (sec) <<\r' % (time.time()-start_time),)
                sys.stdout.flush()
                train_loss.append(loss)
            train_loss=np.mean(train_loss,dtype=float)
            print( 'Train Epoch %d loss %f' % (epoch, train_loss))

            # valid
            valid_loss = []
            valid_pred = []
            for i in range(0, len(X_valid), config.batch_size):
                input_x = X_valid[slice(i, i + config.batch_size)]
                dict_X = dict_valid[slice(i, i + config.batch_size)]
                input_x = padding3(input_x, word2id[PAD])
                dict_X = padding2(dict_X, word2id[PAD])
                y = y_valid[slice(i, i + config.batch_size)]
                y = padding(y, 3)
                loss, predict = model.dev_step(sess, input_x, dict_X, y)
                valid_loss.append(loss)
                valid_pred += predict
            valid_loss = np.mean(valid_loss, dtype=float)
            P, R, F = evaluate_word_PRF(valid_pred, y_valid)
            print( 'Valid Epoch %d loss %f' % (epoch, valid_loss))
            print( 'P:%f R:%f F:%f' % (P, R, F))


            if F>best_f1:
                best_f1=F
                best_e=0
                saver.save(sess,checkpoints_model)
            else:
                best_e+=1

            test_pred = []
            for i in range(0, len(x_test), config.batch_size):
                input_x = x_test[slice(i, i + config.batch_size)]
                dict_X = dict_test[slice(i, i + config.batch_size)]
                input_x = padding3(input_x, word2id[PAD])
                dict_X = padding2(dict_X, word2id[PAD])
                predict= model.predict_step(sess, input_x,dict_X)
                test_pred += predict
            P, R, F = evaluate_word_PRF(test_pred, y_test)
            print( 'Test: P:%f R:%f F:%f Best_F:%f' % (P, R, F,best_f1))
            print( '--------------------------------')


            if best_e>3000:
                print( 'Early stopping')
                break

        print( 'best_f1 on validset is %f' % best_f1)

def predict():
    if FLAGS.model_path==None:
        raise 'Model path is None!'
    word2id = get_word2id(train_data_path, bigram_words=bigram_words_path,min_bw_frequence=FLAGS.min_bg_freq)
    id2word=build_reverse_dictionary(word2id)
    x_test, y_test = get_train_data(test_data_path, word2id)
    dict_test = generate_dicttag(test_data_path, general_words_path=FLAGS.general,domain_words_path=FLAGS.domain)
    init_embedding = None

    print( 'test_data_path: %s' % test_data_path)
    print( 'bigram_words_path: %s' % bigram_words_path)
    print( 'model_path: %s' % FLAGS.model_path)
    print( 'min_bg_freq: %d' % FLAGS.min_bg_freq)

    with tf.Session() as sess:
        model=getattr(models,FLAGS.model)(vocab_size=len(word2id),word_dim=config.word_dim,hidden_dim=config.hidden_dim,
                    pad_word=word2id[PAD],init_embedding=init_embedding,num_classes=config.num_classes,clip=config.clip,
                    lr=config.lr,l2_reg_lamda=config.l2_reg_lamda,num_layers=config.num_layers,rnn_cell=config.rnn_cell,
                    bi_direction=config.bi_direction,hidden_dim2=config.hidden_dim2,hyper_embedding_size=config.hyper_embed_size)
        checkpoints_model = os.path.join('checkpoints',FLAGS.model_path)
        saver = tf.train.Saver(tf.all_variables())
        ckpt = tf.train.get_checkpoint_state(checkpoints_model)
        if ckpt and ckpt.model_checkpoint_path:
            print( 'test_start!')
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print( '没有训练好的模型')
            exit()
        test_pred=[]
        for i in range(0,len(x_test),config.batch_size):
            input_x=x_test[slice(i,i+config.batch_size)]
            dict_X = dict_test[slice(i, i + config.batch_size)]
            input_x=padding3(input_x,word2id[PAD])
            dict_X = padding2(dict_X, word2id[PAD])
            predict = model.predict_step(sess, input_x, dict_X)
            test_pred+=predict
        P,R,F=evaluate_word_PRF(test_pred,y_test)
        print( '%s: P:%f R:%f F:%f' % (FLAGS.model_path,P,R,F))
        print( '------------------------------------------')


if __name__ == '__main__':
    if FLAGS.is_train:
        train()
        
    else:
        predict()










