import numpy as np
import theano
import theano.tensor as T

class softmax(object):
    def __init__(self,n_input,n_output,x):
        self.n_input=n_input
        self.n_output=n_output

        self.logit_shape=x.shape
        self.x=x.reshape([self.logit_shape[0]*self.logit_shape[1],self.logit_shape[2]])

        init_W=np.asarray(np.random.uniform(low=-np.sqrt(1./n_input),
                                             high=np.sqrt(1./n_input),
                                             size=(n_input,n_output)),dtype=theano.config.floatX)
        init_b=np.zeros((n_output),dtype=theano.config.floatX)

        self.W=theano.shared(value=init_W,name='output_W')
        self.b=theano.shared(value=init_b,name='output_b')

        self.params=[self.W,self.b]

        self.build()

    def build(self):

        self.activation=T.nnet.softmax(T.dot(self.x,self.W)+self.b)
        self.predict=T.argmax(self.activation,axis=-1)
        
        
class adaptive_softmax():
    
    def __init__(self, inputs, labels, y_mask,
                 n_dim,
                 cutoff, project_factor=4):
        '''
        Args:
            inputs: flattened logits with shape of [n_step*n_batch, n_dim]
            labels: flattened labels with shape of [n_step*n_batch]
            y_mask: mask the null space of sentences with shape of [n_step*n_batch]
            cutoff: frequency binning, i.e. [2000, vocab_size]
            project_factor: project for low-frequency words
        '''
        self.input_dim = n_dim
        self.sample_num = inputs.shape[0]
        self.cluster_num = len(cutoff) - 1
        self.head_dim = cutoff[0] + self.cluster_num
        self.params = []
        self.y_mask = y_mask

        init_head_w = np.asarray(np.random.uniform(low=-np.sqrt(1./self.input_dim),
                                              high=np.sqrt(1./self.input_dim),
                                              size=(self.input_dim,self.head_dim)))
        self.head_w=theano.shared(value=init_head_w,name='head_w')
        self.params.append(self.head_w)

        tail_project_factor = project_factor
        tail_w_list = []
        for i in range(self.cluster_num):
            project_dim = max(1, self.input_dim // tail_project_factor)
            tail_dim = cutoff[i + 1] - cutoff[i]
            _tail_proj_w = np.asarray(np.random.uniform(low=-np.sqrt(1./self.input_dim),
                                             high=np.sqrt(1./self.input_dim),
                                             size=(self.input_dim, project_dim)),dtype=theano.config.floatX)
            _tail_w = np.asarray(np.random.uniform(low=-np.sqrt(1./project_dim),
                                             high=np.sqrt(1./project_dim),
                                             size=(project_dim,tail_dim)),dtype=theano.config.floatX)
            tail_proj_w = theano.shared(value=_tail_proj_w, name="adaptive_softmax_tail{}_proj_w".format(i+1))
            tail_w = theano.shared(value=_tail_w, name="adaptive_softmax_tail{}_w".format(i+1))
            tail_w_list.append([tail_proj_w, tail_w])
            tail_project_factor *= project_factor
            self.params.append(tail_proj_w)
            self.params.append(tail_w)

        training_losses = []
        loss = 0.
        head_labels = labels
        for i in range(self.cluster_num):
            mask = T.bitwise_and(T.ge(labels, cutoff[i]), T.lt(labels, cutoff[i + 1]))  # mask words not in this cluster

            # update head labels with mask (we take words with high frequency as head part)
            head_labels = T.switch(mask, T.constant([cutoff[0] + i]).repeat(self.sample_num), head_labels)
            # we take words with low frequency as a unified label, and append to tail of head labels
            # i.g. 3000 -> 2000, 2001 -> 2000
            # head labels: [0,1,2...,1999] + [2000]

            # compute tail loss
            # first remove the words not in this cluster (this range of frequency) with mask
            tail_inputs = inputs[mask.nonzero()]
            # encode on tail inputs and get logits
            tail_logits = T.dot(T.dot(tail_inputs, tail_w_list[i][0]), tail_w_list[i][1])
            # update tail labels, relabel by (- 2000)
            tail_labels = (labels - cutoff[i])[mask.nonzero()]
            # y_mask that eases the effect of null space in the tail of sentences
            tail_y_mask = self.y_mask[mask.nonzero()]
            tail_logits = tail_logits[T.eq(tail_y_mask, 1).nonzero()]
            tail_labels = tail_labels[T.eq(tail_y_mask, 1).nonzero()]
            # to solve NaN problem
            tail_logits = T.clip(tail_logits, 1.0e-8, 1.0 - 1.0e-8)
            # tail_loss for words with low frequency
            tail_loss = T.mean(T.nnet.categorical_crossentropy(tail_logits, tail_labels))
            training_losses.append(tail_loss)
            loss += tail_loss
            self.tail_logits = tail_logits
            self.tail_labels = tail_labels
            self.tail_loss = tail_loss

        # compute head loss
        # encode head_inputs
        head_logits = T.dot(inputs, self.head_w)
        # y_mask that eases the effect of null space in the tail of sentences
        head_logits = head_logits[T.eq(self.y_mask, 1).nonzero()]
        head_labels = head_labels[T.eq(self.y_mask, 1).nonzero()]
        # to solve NaN problem
        head_logits = T.clip(head_logits, 1.0e-8, 1.0 - 1.0e-8)
        head_loss = T.mean(T.nnet.categorical_crossentropy(head_logits, head_labels))
        loss += head_loss
        training_losses.append(head_loss)

        self.loss = loss
        self.training_losses = training_losses
        self.head_loss = head_loss

