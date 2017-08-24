import theano
if theano.config.device=='cpu':
    from theano.tensor.shared_randomstreams import RandomStreams
elif theano.config.device=='gpu':
    from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from softmax import softmax, adaptive_softmax
from lstm import LSTM
from updates import *


class RNNLM(object):
    def __init__(self, n_input, n_hidden, n_batch, n_output, optimizer=sgd, p=0.5, use_adaptive_softmax=True):
        self.x = T.imatrix('batched_sequence_x')  # n_batch, maxlen
        self.x_mask = T.matrix('x_mask')
        self.y = T.imatrix('batched_sequence_y')
        self.y_mask = T.matrix('y_mask')

        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        init_Embd = np.asarray(np.random.uniform(low=-np.sqrt(1. / n_output),
                                                 high=np.sqrt(1. / n_output),
                                                 size=(n_output, n_input)),
                               dtype=theano.config.floatX)
        self.E = theano.shared(value=init_Embd, name='word_embedding',borrow=True)

        self.optimizer = optimizer
        self.p = p
        self.is_train = T.iscalar('is_train')
        self.n_batch = n_batch
        self.epsilon = 1.0e-15
        self.rng = RandomStreams(1234)
        self.use_adaptive_softmax = use_adaptive_softmax
        self.build()

    def build(self):
        print 'building rnn cell...'
        hidden_layer = LSTM(self.rng,
                            self.n_input, self.n_hidden, self.n_batch,
                            self.x, self.E, self.x_mask,
                            self.is_train, self.p)
        print 'building softmax output layer...'
        if self.use_adaptive_softmax:
            cutoff = [2000, self.n_output]
            softmax_inputs = hidden_layer.activation
            logit_shape = softmax_inputs.shape
            softmax_inputs = softmax_inputs.reshape([logit_shape[0]*logit_shape[1], logit_shape[2]])
            labels = self.y.flatten()
            y_mask = self.y_mask.flatten()
            output_layer = adaptive_softmax(softmax_inputs, labels, y_mask,
                                            self.n_hidden,
                                            cutoff)
            #cost = T.sum(output_layer.loss)
            training_loss = output_layer.training_losses
            cost = output_layer.loss
        else:
            output_layer = softmax(self.n_hidden, self.n_output, hidden_layer.activation)
            cost = self.categorical_crossentropy(output_layer.activation, self.y)
        self.params = [self.E, ]
        self.params += hidden_layer.params
        self.params += output_layer.params

        lr = T.scalar("lr")
        gparams = [T.clip(T.grad(cost, p), -1, 1) for p in self.params]
        updates = self.optimizer(self.params, gparams, lr)

        self.train = theano.function(inputs=[self.x, self.x_mask, self.y, self.y_mask, lr],
                                     outputs=[cost,hidden_layer.activation, output_layer.head_logits, output_layer.head_labels],
                                     updates=updates,
                                     givens={self.is_train: np.cast['int32'](1)})

        self.test = theano.function(inputs=[self.x, self.x_mask,self.y, self.y_mask],
                                    outputs=cost,
                                    givens={self.is_train: np.cast['int32'](0)})

    def categorical_crossentropy(self, y_pred, y_true):
        y_pred = T.clip(y_pred, self.epsilon, 1.0 - self.epsilon)
        y_true = y_true.flatten()
        nll = T.nnet.categorical_crossentropy(y_pred, y_true)
        return T.sum(nll * self.y_mask.flatten())
