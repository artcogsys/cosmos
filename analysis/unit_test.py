import unittest

import numpy as np

from chainer import Chain
import chainer.links as L
import chainer.functions as F

from chainer.datasets import TupleDataset

from supervised_learning.agents import SupervisedAgent
from supervised_learning.models import *
from supervised_learning.world import World
from supervised_learning.iterators import RandomIterator, SequentialIterator

import matplotlib.pyplot as plt

class UnitTest(unittest.TestCase):

    def test_supervised_monitor(self):

        n_epochs = 5

        # create toy data - predict if the sum of the previous input is larger than 1.0
        def create_data():
            X = np.random.rand(1000, 2).astype('float32')
            T = (np.sum(X, 1) > 1.0)
            T = np.hstack([0, T[1:]]).astype('int32')
            data = TupleDataset(X, T)
            return SequentialIterator(data, batch_size=20)

        train_iter = create_data()
        test_iter = create_data()

        # define network - saves hidden states after each call
        class RNN(Chain):

            def __init__(self, n_input=None, n_output=1, n_hidden=10):
                super(RNN, self).__init__(lstm=L.LSTM(n_input, n_hidden), fc=L.Linear(n_hidden, n_output))

                self.state = []

            def __call__(self, x):

                y = self.lstm(x)
                self.state.append(y.data)
                return self.fc(y)

            @property
            def has_state(self):
                return True

            def reset_state(self):
                self.lstm.reset_state()

        # define network
        net = RNN(n_output=2, n_hidden=3)

        # define agent
        agent = SupervisedAgent(Classifier(net, gpu=-1), chainer.optimizers.Adam(), cutoff=train_iter.n_batches)

        # define world
        world = World(agent)

        # run world in training mode
        train_loss, test_loss = world.train(train_iter, n_epochs=n_epochs, test_iter=test_iter)

        # convert epoch x batch x variables to time x variables
        data = np.asarray(net.state)
        data = data.reshape([np.prod(data.shape[:2])] + list(data.shape[2:]), order='F')

        plt.plot(data[:1000])
        plt.show()


if __name__ == '__main__':
    unittest.main()
