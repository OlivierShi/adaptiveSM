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
    
    def __init__(self, n_input, n_output, x, cutoff, project_factor=4):
        
