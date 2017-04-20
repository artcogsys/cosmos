import unittest

from chainer.datasets import TupleDataset

import networks
from agents import *
from models import *
from world import World
from iterators import RandomIterator, SequentialIterator

class UnitTest(unittest.TestCase):

    def test_stateless_network(self):

        n_epochs = 20

        # create toy data - predict if the sum of the input is larger than 1.0
        def create_data():
            X = np.random.rand(1000, 2).astype('float32')
            T = (np.sum(X, 1) > 1.0).astype('int32')
            data = TupleDataset(X, T)
            return RandomIterator(data, batch_size=20)

        train_iter = create_data()
        test_iter = create_data()

        # define network
        net = networks.MLP(n_output=2, n_hidden=3)

        # define agent
        agent = StatelessAgent(Classifier(net), chainer.optimizers.Adam(), gpu=-1)

        # define world
        world = World(agent)

        # run world in training mode
        train_loss, test_loss = world.train(train_iter, n_epochs=n_epochs, test_iter=test_iter)

        print test_loss

    def test_stateful_network(self):

        n_epochs = 20

        # create toy data - predict if the sum of the previous input is larger than 1.0
        def create_data():
            X = np.random.rand(1000, 2).astype('float32')
            T = (np.sum(X, 1) > 1.0)
            T = np.hstack([0, T[1:]]).astype('int32')
            data = TupleDataset(X, T)
            return SequentialIterator(data, batch_size=20)

        train_iter = create_data()
        test_iter = create_data()

        # define network
        net = networks.RNN(n_output=2, n_hidden=3)

        # define agent
        agent = StatefulAgent(Classifier(net), chainer.optimizers.Adam(), cutoff=train_iter.n_batches, gpu=-1)

        # define world
        world = World(agent)

        # run world in training mode
        train_loss, test_loss = world.train(train_iter, n_epochs=n_epochs, test_iter=test_iter)

        print test_loss


if __name__ == '__main__':
    unittest.main()
