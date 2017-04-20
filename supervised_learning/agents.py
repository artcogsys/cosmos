import numpy as np

import chainer
from chainer import Variable
from chainer import cuda

class Agent(object):

    def __init__(self, model, optimizer, gpu=-1):

        self.model = model

        self.optimizer = optimizer
        self.optimizer.setup(self.model)

        if gpu>=0:
            self.xp = cuda.cupy
            chainer.cuda.get_device(gpu).use()
            self.model.to_gpu()
        else:
            self.xp = np

    def __call_(self, data):
        """
        Runs networks in forward mode and applies optional output function
        :param data:
        :return: post-processed output
        """

        return self.model.predict(data)

    def train(self, data):
        raise NotImplementedError

    def test(self, data):
        """
        Returns the loss for one batch
        :param data:
        :return: loss
        """

        loss = self.model(map(lambda x: Variable(self.xp.asarray(x)), data))

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
        loss = self.model(map(lambda x: Variable(self.xp.asarray(x)), data))
        loss.backward()
        self.optimizer.update()

        return loss.data

class StatefulAgent(Agent):

    def __init__(self, model, optimizer, cutoff, gpu=-1):

        super(StatefulAgent, self).__init__(model, optimizer, gpu)

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

        _loss = self.model(map(lambda x: Variable(self.xp.asarray(x)), data))
        self.loss += _loss

        if self.counter % self.cutoff == 0:

            self.optimizer.zero_grads()
            self.loss.backward()
            self.loss.unchain_backward()
            self.optimizer.update()

            self.loss = Variable(self.xp.zeros((), 'float32'))

        return _loss.data

    def reset(self):
        self.model.reset()
        self.counter = 0
