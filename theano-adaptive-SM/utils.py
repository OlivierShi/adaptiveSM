import numpy as np
import cPickle as pickle

def save_model(f,model):
    ps={}
    for p in model.params:
        ps[p.name]=p.get_value()
    pickle.dump(ps,open(f,'wb'))

def load_model(f,model):
    ps=pickle.load(open(f,'rb'))
    for p in model.params:
        p.set_value(ps[p.name])
    return model


class Vocab(object):
    def __init__(self, vocab_size):
        self.word_count = {'</s>': 0, '<unk>': 0, '<s>': 0}
        self.words = None
        self.vocab_size = vocab_size

    def build_from_files(self, files):
        if type(files) is not list:
            raise ValueError("buildFromFiles input type error")

        print ("build vocabulary from files ...")
        for _file in files:
            line_num = 0
            for line in open(_file):
                line_num += 1
                for w in line.strip().split():
                    if w in self.word_count:
                        self.word_count[w] += 1
                    else:
                        self.word_count[w] = 1
            # self.word_count['<s>'] += line_num
            self.word_count['</s>'] += line_num
        count_pairs = sorted(self.word_count.items(), key=lambda x: (-x[1], x[0]))
        self.words, counts = list(zip(*count_pairs))
        self.word2id = dict(zip(self.words, range(len(self.words))))

        self.UNK_ID = self.word2id['<unk>']
        print ("vocab size: {}".format(self.size()))

    def encode(self, sentence):
        return [self.word2id[w] if self.word2id.has_key(w) else self.UNK_ID for w in sentence]

    def decode(self, ids):
        return [self.words[_id] for _id in ids]

    def size(self):
        return len(self.words)

    def word2id(self):
        return self.word2id

class TextIterator:
    def __init__(self,source,is_train,n_batch,maxlen=None,n_words_source=-1,vocab_size=None):
        self.file = source
        self.source=open(source,'r')
        self.n_batch=n_batch
        self.maxlen=maxlen
        self.n_words_source=n_words_source
        self.end_of_data=False
        self.vocab_size = vocab_size
        if is_train:
            self.vocab = Vocab(self.vocab_size)
            self.vocab.build_from_files([self.file])
            self.word_dict = self.vocab.word2id

    def __iter__(self):
        return self

    def reset(self):
        self.source.seek(0)

    def set_word_dict(self, word_dict):
        self.word_dict = word_dict

    def get_word_dict(self):
        return self.vocab.word2id

    def next(self):
        if self.end_of_data:
            self.end_of_data = False
            self.reset()
            raise StopIteration
        data = []
        bos_id = self.word_dict['<s>']
        unk_id = self.word_dict['<unk>']
        eos_id = self.word_dict['</s>']
        try:
            while True:
                line = self.source.readline()
                if line == '':
                    raise IOError
                tokens = line.strip().split()
                tokens2id = [bos_id] + \
                            [self.word_dict[w] if self.word_dict.has_key(w) else unk_id for w in tokens] + \
                            [eos_id]
                if self.maxlen and len(line) > self.maxlen:
                    continue
                data.append(tokens2id)
                if len(data) >= self.n_batch:
                    break
        except IOError:
            self.end_of_data = True

        if len(data) <= 0:
            self.end_of_data = False
            self.reset()
            raise StopIteration
        return prepare_data(data)

def prepare_data(seqs_x):
    lengths_x=[len(s)-1 for s in seqs_x]
    n_samples=len(seqs_x)
    maxlen_x=np.max(lengths_x)

    x=np.zeros((maxlen_x,n_samples)).astype('int32')
    y=np.zeros((maxlen_x,n_samples)).astype('int32')
    x_mask=np.zeros((maxlen_x,n_samples)).astype('float32')
    y_mask=np.zeros((maxlen_x,n_samples)).astype('float32')

    for idx,s_x in enumerate(seqs_x):
        x[:lengths_x[idx],idx]=s_x[:-1]
        y[:lengths_x[idx],idx]=s_x[1:]
        x_mask[:lengths_x[idx],idx]=1
        y_mask[:lengths_x[idx],idx]=1

    return x,x_mask,y,y_mask


# train_data=TextIterator('ptb_data/ptb.train.txt', is_train=True,n_batch=100)
# i = 0
# for x,x_mask,y,y_mask in train_data:
#     i += 1
#     print i
