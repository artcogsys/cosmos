import unittest

import numpy as np

import networks
from agents import *
from models import *
from world import World
from tasks import EvidenceTask

class UnitTest(unittest.TestCase):

    def test_reinforce_stateless(self):
        """
        Test REINFORCE on stateless network
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

    def test_reinforce_stateful(self):
        """
        Test REINFORCE on stateful network
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

    def test_aac_stateful(self):
        """
        Test Advantage Actor-Critic on stateful network
        """

        n_steps = 100

        task = EvidenceTask()

        # define network
        net = networks.RNN(n_output=3, n_hidden=3)

        # define agent
        agent = AACAgent(ActorCriticModel(net, gpu=-1), chainer.optimizers.Adam())

        # define world
        world = World(agent)

        # run world in training mode
        rewards = world.train(task, n_steps=n_steps)

if __name__ == '__main__':
    unittest.main()
