import time

from rnnlm import *
from utils import TextIterator,save_model
from updates import *
import logging
from argparse import ArgumentParser 
import sys

lr=0.01
p=0.5
NEPOCH=200

n_input=256
n_hidden=256
optimizer=sgd

argument = ArgumentParser(usage='it is usage tip', description='no')  
argument.add_argument('--train_file', default='ptb_data/ptb.train.txt', type=str, help='train dir')
argument.add_argument('--valid_file', default='ptb_data/ptb.valid.txt', type=str, help='valid dir')
argument.add_argument('--test_file', default='ptb_data/ptb.test.txt', type=str, help='test dir')
argument.add_argument('--vocab_size', default=10001, type=int, help='vocab size')
argument.add_argument('--batch_size', default=10, type=int, help='batch size')


args = argument.parse_args()  


train_datafile=args.train_file
valid_datafile=args.valid_file
test_datafile=args.test_file
n_batch=args.batch_size
vocabulary_size=args.vocab_size
n_words_source=-1
disp_freq=40
valid_freq=1000
test_freq=2000
save_freq=20000
clip_freq=2000
pred_freq=20000

def evaluate(test_data,model):
    cost=0.
    index=0.
    for x,x_mask,y,y_mask in test_data:
        if x.shape[1] is not n_batch:
            continue
        index+=1.
        cost+=model.test(x,x_mask,y,y_mask)
    return cost/index

def train(lr):
    # Load data
    print 'loading dataset...'

    train_data=TextIterator(train_datafile,is_train=True,n_words_source=n_words_source,n_batch=n_batch,vocab_size=vocabulary_size)
    word_dict = train_data.get_word_dict()
    valid_data=TextIterator(valid_datafile,is_train=False,n_words_source=n_words_source,n_batch=n_batch)
    valid_data.set_word_dict(word_dict)
    test_data=TextIterator(test_datafile,is_train=False,n_words_source=n_words_source,n_batch=n_batch)
    test_data.set_word_dict(word_dict)
    print 'building model...'
    model=RNNLM(n_input,n_hidden,n_batch,vocabulary_size,optimizer,p)
    print 'training start...'
    start=time.time()
    idx=0
    for epoch in xrange(NEPOCH):
        error=0.
        in_start=time.time()
        for x,x_mask,y,y_mask in train_data:
            if x.shape[1] is not n_batch:
                continue
            idx+=1
            beg_time=time.time()
            train_out=model.train(x,x_mask,y,y_mask,lr)
            cost = train_out[0]
            head_cost = train_out[2]
            tail_cost = train_out[3]
            error+=np.sum(cost)
            if np.isnan(cost) or np.isinf(cost):
                print 'NaN Or Inf detected!'
                return -1
            if idx % disp_freq==0:
                print 'time: ',time.time()-beg_time,'epoch:',epoch,'idx:',idx,'cost:',error/disp_freq,'ppl:',np.exp(error/disp_freq),'lr:',lr
                print "Head loss: ", head_cost
                print "Tail loss: ", tail_cost
                error=0
            if idx%save_freq==0:
                print 'dumping...'
                save_model('./model/parameters_%.2f.pkl'%(time.time()-start),model)
            if idx % valid_freq==0:
                print 'validing....'
                valid_cost=evaluate(valid_data,model)
                print 'valid_cost:',valid_cost,'perplexity:',np.exp(valid_cost)
            if idx % test_freq==0:
                print 'testing...'
                test_cost=evaluate(test_data,model)
                print 'test cost:',test_cost,'perplexity:',np.exp(test_cost)
            if idx%clip_freq==0 and lr >=1e-3:
                lr=lr*0.7
                print 'cliping learning rate:', lr
        print "epoch: ", epoch, "time consuming: ", time.time()-in_start
        sys.stdout.flush()

    print "Finished. Time = "+str(time.time()-start)


if __name__ == '__main__':
    train(lr=lr)
