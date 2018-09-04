"""
network.py
~~~~~~~~~~
A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""

#### Libraries
# Standard library
import random

# Third-party libraries
import numpy as np
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt
from activations.relu import Relu
from activations.leaky_relu import LRelu
from activations.sigmoid import Sigmoid
from load_story import *
import os

class ResiNetwork(object):

    def __init__(self, sizes, activation):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.resi = [np.zeros((y, 1)) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
        self.activation = activation
        self.FORGET_RATE = 0.9
        self.LEARNING_RATE = 0.2

    def feedforward(self, xs, q):
        """Return the output of the network if ``a`` is input."""
        self.load_memory(xs)
        for b, w, r in zip(self.biases, self.weights, self.resi):
            q = self.activation.forward(np.dot(w, q) + b + r)
        self.clear_memory()
        return q

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))


    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for xs, q, y in mini_batch:
            self.load_memory(xs)
            delta_nabla_b, delta_nabla_w = self.backprop(q, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            self.clear_memory()
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def load_memory(self, mem):
        for x in mem:
            i = 0
            for b, w, r in zip(self.biases, self.weights, self.resi):
                z = np.dot(w, x) + b + r * self.FORGET_RATE
                x = self.activation.forward(z)
                self.resi[i] = r * self.FORGET_RATE + x * self.LEARNING_RATE
                i += 1

    def clear_memory(self):
        self.resi = [np.zeros((y, 1)) for y in self.sizes[1:]]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w, r in zip(self.biases, self.weights, self.resi):
            z = np.dot(w, activation) + b + r * self.FORGET_RATE
            zs.append(z)
            activation = self.activation.forward(z)
            activations.append(activation)

        l = self.cost_derivative(activations[-1], y)
        ab = self.activation.backward(zs[-1])
        # backward pass
        delta = l * ab

        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = self.activation.backward(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        correct = 0
        for (xs, q, y) in test_data:
            pred_y = self.feedforward(xs, q)
            # print(np.around(pred_y, 3), y)
            if np.argmax(pred_y) == np.argmax(y):
                correct += 1
        return correct

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)

def generate_data(seed, size):
    np.random.seed(seed)
    X, y = datasets.make_moons(size, noise=0.20)
    ds = []
    for i in range(size):
        if y[i] == 0:
            t = np.array([1, 0])
            t = t.reshape(-1, 1)
            tx = np.array(X[i])
            tx = tx.reshape(-1, 1)
            rx = [tx]
            ds.append((rx, t))
        else:
            t = np.array([0, 1])
            t = t.reshape(-1, 1)
            tx = np.array(X[i])
            tx = tx.reshape(-1, 1)
            rx = [tx]
            ds.append((rx, t))
    return ds

def main():
    challenge = 'data/en/qa3_three-supporting-facts_{}.txt'
    with open(challenge.format('train')) as train_f:
        train = get_stories(train_f)
    with open(challenge.format('test')) as test_f:
        test = get_stories(test_f)

    flatten = lambda data: reduce(lambda x, y: x + y, data)
    vocab = set()
    for story, q, answer in train + test:
        vocab |= set(flatten(story) + q + [answer])
    vocab = sorted(vocab)
    print(vocab)

    # Reserve 0 for masking via pad_sequences
    vocab_size = len(vocab) + 1
    word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
    story_maxlen = max(map(len, (x for x, _, _ in train + test)))
    query_maxlen = max(map(len, (x for _, x, _ in train + test)))
    if story_maxlen > query_maxlen:
        query_maxlen = story_maxlen
    else:
        story_maxlen = query_maxlen

    x, xq, y = vectorize_stories(train, word_idx, story_maxlen, query_maxlen)
    tx, txq, ty = vectorize_stories(test, word_idx, story_maxlen, query_maxlen)

    print('vocab = {}'.format(vocab))
    print('x.shape = {}'.format(x.shape))
    print('xq.shape = {}'.format(xq.shape))
    print('y.shape = {}'.format(y.shape))
    print('story_maxlen, query_maxlen = {}, {}'.format(story_maxlen, query_maxlen))
    print(x[0].shape)
    print(xq[0].shape)
    print(y[0].shape)
    # exit()

    train_ds = list(zip(x, xq, y))
    test_ds = list(zip(tx, txq, ty))
    # train_ds = generate_data(0, 200)
    # test_ds = generate_data(1, 20)
    #
    # # print(train_ds)
    model = ResiNetwork([len(x[0][0]), 500, 250, len(y[0])], Relu())
    model.SGD(train_ds, 1000, 10, 0.5, test_ds)

if __name__ == "__main__":
    main()
