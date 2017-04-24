import unittest

from chainer.datasets import TupleDataset

import numpy as np

import networks
from agents import REINFORCEAgent
from models import *
from world import World
from tasks import EvidenceTask

class UnitTest(unittest.TestCase):

    def test_stateless_network(self):
        """
        Test training procedure for stateless network
        """

        n_steps = 100

        task = EvidenceTask()

        # define network
        net = networks.MLP(n_output=2, n_hidden=3)

        # define agent
        agent = REINFORCEAgent(ActorModel(net, gpu=-1), chainer.optimizers.Adam())

        # define world
        world = World(agent)

        # run world in training mode
        rewards = world.train(task, n_steps=n_steps)

    def test_stateful_network(self):
        """
        Test training procedure for stateful network
        """

        n_steps = 100

        task = EvidenceTask()

        # define network
        net = networks.RNN(n_output=2, n_hidden=3)

        # define agent
        agent = REINFORCEAgent(ActorModel(net, gpu=-1), chainer.optimizers.Adam())

        # define world
        world = World(agent)

        # run world in training mode
        rewards = world.train(task, n_steps=n_steps)

if __name__ == '__main__':
    unittest.main()
