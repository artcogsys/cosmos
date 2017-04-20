import numpy as np

import chainer.functions as F
from chainer import Variable
from chainer import cuda

class Agent(object):

    def __init__(self, net, optimizer, gpu=-1, loss_function=F.MeanSquaredError, output_function=lambda x: x):

        self.net = net

        self.optimizer = optimizer
        self.optimizer.setup(self.net)

        self.xp = np if gpu == -1 else cuda.cupy

        self.loss_function = loss_function
        self.output_function = output_function

    def __call_(self, data):
        """
        Runs networks in forward mode and applies optional output function
        :param data:
        :return: post-processed output
        """

        return self.output_function(self.net(Variable(data[0])))

    def train(self, data):
        raise NotImplementedError

    def test(self, data):
        """
        Returns the loss for one batch
        :param data:
        :return: loss
        """

        loss = self.loss_function(self.net(Variable(data[0])), Variable(data[1]))

        return loss.data

    def reset(self):
        """
        Stateless agents don't require a state reset
        :return:
        """

        pass

class StatelessAgent(Agent):

    def train(self, data):
        """
        Trains on one batch and returns the loss
        :param data:
        :return: loss
        """

        self.optimizer.zero_grads()  # zero the gradient buffer
        loss = self.loss_function(self.net(Variable(self.xp.asarray(data[0]))), Variable(self.xp.asarray(data[1])))
        loss.backward()
        self.optimizer.update()

        return loss.data

class StatefulAgent(Agent):

    def __init__(self, net, optimizer, cutoff, gpu=-1, loss_function=F.MeanSquaredError, output_function=lambda x: x):

        super(StatefulAgent, self).__init__(net, optimizer, gpu, loss_function, output_function)

        self.cutoff = cutoff
        self.counter = 0

        self.loss = Variable(self.xp.zeros((), 'float32'))

    def train(self, data):
        """
        Trains on one batch and returns the loss
        :param data: 
        :return: loss
        """

        self.counter += 1

        _loss = self.loss_function(self.net(Variable(self.xp.asarray(data[0]))), Variable(self.xp.asarray(data[1])))
        self.loss += _loss

        if self.counter % self.cutoff == 0:

            self.optimizer.zero_grads()
            self.loss.backward()
            self.loss.unchain_backward()
            self.optimizer.update()

            self.loss = Variable(self.xp.zeros((), 'float32'))

        return _loss.data

    def reset(self):
        self.net.reset()
        self.counter = 0
