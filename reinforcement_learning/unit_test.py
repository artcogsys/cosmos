import unittest

import chainer
from chainer.datasets import TupleDataset

from reinforcement_learning import networks
from reinforcement_learning.agents import *
from reinforcement_learning.world import World
from reinforcement_learning.iterators import FooTask

class UnitTest(unittest.TestCase):

    def test_stateless_network(self):

        n_epochs = 20

        train_iter = FooTask(batch_size=1, n_batches=10**4)
        test_iter = FooTask(batch_size=1, n_batches=10**3)

        # define network
        net = networks.MLP(n_output=train_iter.n_output, n_hidden=5)

        # define agent
        agent = REINFORCEAgent(net, chainer.optimizers.Adam())

        # define world
        world = World(agent)

        # run world in training mode
        train_loss, test_loss = world.train(train_iter, n_epochs=n_epochs, test_iter=test_iter)

    def atest_stateful_network(self):

        n_epochs = 20
        batch_size = 20

        # create toy data - predict if the sum of the input is larger than 1.0
        def create_data():
            X = np.random.rand(1000, 2).astype('float32')
            T = (np.sum(X, 1) > 1.0)
            T = np.hstack([0, T[1:]]).astype('int32')
            data = TupleDataset(X, T)
            return SequentialIterator(data, batch_size=batch_size)

        train_iter = create_data()
        test_iter = create_data()

        # define network
        net = networks.RNN(n_input=2, n_output=2, n_hidden=3)

        # define agent
        agent = StatefulAgent(net, chainer.optimizers.Adam(), cutoff=train_iter.n_batches,
                              loss_function=F.softmax_cross_entropy, output_function=F.softmax)

        # define world
        world = World(agent)

        # run world in training mode
        train_loss, test_loss = world.train(train_iter, n_epochs=n_epochs, test_iter=test_iter)


if __name__ == '__main__':
    unittest.main()
