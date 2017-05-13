import unittest

import numpy as np

from chainer import Chain
import chainer.links as L
import chainer.functions as F

from chainer.datasets import TupleDataset

from supervised_learning.agents import SupervisedAgent
from supervised_learning.models import *
from supervised_learning.world import World as SLWorld
from supervised_learning.iterators import RandomIterator, SequentialIterator

from reinforcement_learning.agents import *
from reinforcement_learning.models import *
from reinforcement_learning.world import World as RLWorld
from reinforcement_learning.tasks import EvidenceTask

from analysis.base import *

class UnitTest(unittest.TestCase):

    def test_supervised_monitor(self):

        n_epochs = 20

        # create toy data - predict if the sum of the input is larger than 1.0
        def create_data():
            X = np.random.rand(1000, 2).astype('float32')
            T = (np.sum(X, 1) > 1.0).astype('int32')
            data = TupleDataset(X, T)
            return RandomIterator(data, batch_size=20)

        train_iter = create_data()
        test_iter = create_data()

        # define network - saves hidden states after each call
        class MLP(Chain):
            def __init__(self, n_input=None, n_output=1, n_hidden=10):
                super(MLP, self).__init__(l1=L.Linear(n_input, n_hidden), l2=L.Linear(n_hidden, n_output))

                self.hidden = []

            def __call__(self, x):
                h = F.relu(self.l1(x))
                self.hidden.append(h.data)
                return self.l2(h)

            @property
            def has_state(self):
                return False

        net = MLP(n_output=2, n_hidden=3)

        # define agent
        agent = SupervisedAgent(Classifier(net, gpu=-1), chainer.optimizers.Adam())

        # define world
        world = SLWorld(agent)

        # run world in training mode
        train_loss, test_loss = world.train(train_iter, n_epochs=n_epochs, test_iter=test_iter)


if __name__ == '__main__':
    unittest.main()
