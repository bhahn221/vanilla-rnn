# replication of https://github.com/dennybritz/rnn-tutorial-rnnlm

import numpy as np
from ptb import *

import operator
from datetime import datetime, date, time
import sys

import matplotlib.pyplot as plt

def softmax(x):
    xt = np.exp(x - np.max(x))
    
    return xt / np.sum(xt)

class RNNNumpy:
    def __init__(self, word_dim, hidden_dim=100, bptt_truncate=4):
        # assign instance variables
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate

        # randomly initialize the network parameters
        self.U = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, word_dim))
        self.V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (word_dim, hidden_dim))
        self.W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim))

def forward_propagation(self, x):
    # the total number of time steps
    T = len(x)

    # NOTE during forward propagation we save all hidden states in s because need them later.
    # we add one additional element for the initial hidden, which we set to 0
    s = np.zeros((T + 1, self.hidden_dim))
    s[-1] = np.zeros(self.hidden_dim)

    # NOTE the outputs at each time step. again, we save them for later
    o = np.zeros((T, self.word_dim))

    # for each time step...
    for t in np.arange(T):
        # note that we are indexing U by x[t], this is the same as multiplying U with a one_hot vector.
        s[t] = np.tanh(self.U[:,x[t]] + self.W.dot(s[t-1]))
        o[t] = softmax(self.V.dot(s[t]))

    return [o, s]

RNNNumpy.forward_propagation = forward_propagation

def predict(self, x):
    # perform forward propagation and return index of the highest score
    o, s = self.forward_propagation(x)

    return np.argmax(o, axis=1)

RNNNumpy.predict = predict

def calculate_total_loss(self, x, y):
    L = 0
    
    # for each sentence...
    for i in np.arange(len(y)):
        o, s = self.forward_propagation(x[i])

        # we only care about our prediction of the "correct" words
        correct_word_predictions = o[np.arange(len(y[i])), y[i]]

        # add to the loss based on how off we were
        L += -1 * np.sum(np.log(correct_word_predictions))

    return L

def calculate_loss(self, x, y):
    #divide the total loss by the number of training examples
    N = np.sum((len(y_i) for y_i in y))

    return self.calculate_total_loss(x, y) / N

RNNNumpy.calculate_total_loss = calculate_total_loss
RNNNumpy.calculate_loss = calculate_loss

def bptt(self, x, y):
    T = len(y)

    # perform forward propagation
    o, s = self.forward_propagation(x)

    # we accumulate the gradients in these variables
    dLdU = np.zeros(self.U.shape)
    dLdV = np.zeros(self.V.shape)
    dLdW = np.zeros(self.W.shape)
    delta_o = o
    delta_o[np.arange(len(y)), y] -= 1.

    # for each output backwards...
    for t in np.arange(T)[::-1]:
        dLdV += np.outer(delta_o[t], s[t].T)

        # initial delta calculation
        delta_t = self.V.T.dot(delta_o[t]) * (1 - (s[t] ** 2))

        # backpropagation through time (for at most self.bptt_truncate steps)
        for bptt_step in np.arange(max(0, t-self.bptt_truncate), t+1)[::-1]:
            # print "backpropagation step t=%d bptt step=%d " % (t, bptt_step)
            dLdW += np.outer(delta_t, s[bptt_step-1])
            dLdU[:, x[bptt_step]] += delta_t

            # update delta for next step
            delta_t = self.W.T.dot(delta_t) * (1 - s[bptt_step - 1] ** 2)

    return [dLdU, dLdV, dLdW]

RNNNumpy.bptt = bptt

def gradient_check(self, x, y, h=0.001, error_threshold=0.01):
    # calculate the gradients using backpropagation. we want to check if these are correct.
    bptt_gradients = self.bptt(x, y)

    # list of all parameters we want to check.
    model_parameters = ['U', 'V', 'W']

    # gradient check for each parameter
    for pidx, pname in enumerate(model_parameters):
        # get the actual parameter value from the model, e.g. model.W
        parameter = operator.attrgetter(pname)(self)
        print "performing gradient check for parameter %s with size %d." % (pname, np.prod(parameter.shape))
        # iterate ove each element of the parameter matrix, e.g. (0,0), (0,1), ...
        it = np.nditer(parameter, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            ix = it.multi_index

            # save the original value so we can reset it later
            original_value = parameter[ix]

            # estimate the gradient using (f(x+h) - f(x-h)) / (2*h)
            parameter[ix] = original_value + h
            gradplus = self.calculate_total_loss([x], [y])
            parameter[ix] = original_value - h
            gradminus = self.calculate_total_loss([x], [y])
            estimated_gradient = (gradplus - gradminus) / (2*h)

            # reset parameter to original value
            parameter[ix] = original_value

            # the gradient for this parameter calculated using backpropagation
            backprop_gradient = bptt_gradients[pidx][ix]

            # calculate the relativew error: (|x - y| / (|x| + |y|))
            relative_error = np.abs(backprop_gradient - estimated_gradient) / (np.abs(backprop_gradient) + np.abs(estimated_gradient))

            # if the error is too large, fail the gradient check
            if relative_error > error_threshold:
                print "gradient check error: parameter=%s ix=%s" % (pname, ix)
                print "+h loss: %f" % gradplus
                print "-h loss: %f" % gradminus
                print "estimated gradient: %f" % estimated_gradient
                print "backpropagation gradient: %f" % backpropagation_gradient
                print "relative error: %f" % relative_error
                return
            it.iternext()
        print "gradient check for parameter %s passed." % (pname)

RNNNumpy.gradient_check = gradient_check

# performs one step of sgd
def numpy_sgd_step(self, x, y, learning_rate):
    # calculate the gradients
    dLdU, dLdV, dLdW = self.bptt(x, y)

    # change parameters according to gradients and learning rate
    self.U -= learning_rate * dLdU
    self.V -= learning_rate * dLdV
    self.W -= learning_rate * dLdW

RNNNumpy.sgd_step = numpy_sgd_step

# outer sgd Loop
# - model: the rnn model instance
# - x_train: the training data set
# - y_train: the training data labels
# - learning_rate: initial learning rate for sgd
# - nepoch: number of times to iterate through the complete dataset
# - evaluate_loss_after: evaluate the loss after this many epochs
def train_with_sgd(model, x_train, y_train, learning_rate=0.005, nepoch=100, evaluate_loss_after=5):
    # we keep track of the losses so we can plot them later
    losses = []
    num_examples_seen = 0
    for epoch in range(nepoch):
        # optionally evaluate the loss
        if (epoch % evaluate_loss_after == 0):
            loss = model.calculate_loss(x_train, y_train)
            losses.append((num_examples_seen, loss))
            time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print "%s: loss after num_examples_seen=%d epoch=%d: %f" % (time, num_examples_seen, epoch, loss)
            # adjust the learning rate if loss increases
            if (len(losses) > 1 and losses[-1][1] > losses[-2][1]):
                learning_rate = learning_rate * 0.5
                print "setting learning rate to %f" % learning_rate
            sys.stdout.flush()
        
        # for each training example:
        for i in range(len(y_train)):
            # one sgd step
            model.sgd_step(x_train[i], y_train[i], learning_rate)
            num_examples_seen += 1

# to avoid performing millions of expensive calculations we use a smaller vocabulary size for checking
grad_check_vocab_size = 100
np.random.seed(10)
model = RNNNumpy(grad_check_vocab_size, 10, bptt_truncate=1000)
model.gradient_check([0,1,2,3], [1,2,3,4])

# get dataset
train_set, val_set, test_set = load_dataset()
train_x = [s[:-1] for s in train_set]
train_y= [s[1:] for s in train_set]

# train
vocabulary_size = 10001
np.random.seed(10)
model = RNNNumpy(vocabulary_size)
#model.sgd_step(train_x[10], train_y[10], 0.005)
losses = train_with_sgd(model, train_x[:100], train_y[:100], learning_rate=0.005, nepoch=10, evaluate_loss_after=1)

# run RNN
np.random.seed(10)
print(index_to_word(model.predict(train_set[1])))
print(index_to_word(model.predict(train_set[2])))
print(index_to_word(model.predict(train_set[3])))
print(index_to_word(model.predict(train_set[4])))
print(index_to_word(model.predict(train_set[5])))
