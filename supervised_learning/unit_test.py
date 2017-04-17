import unittest

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data.dataset import TensorDataset

import networks
from agents import *
from world import World

class UnitTest(unittest.TestCase):

    def atest_stateless_network(self):

        n_epochs = 30
        batch_size = 20

        # create toy data - predict if the sum of the input is larger than 1.0
        X = np.random.rand(1000, 2)
        T = (np.sum(X, 1) > 1.0).astype('int')
        train_data = TensorDataset(torch.Tensor(X), torch.CharTensor(T))

        # define data loader
        train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

        # define network
        net = networks.MLP(n_input=2, n_output=2, n_hidden=3)

        # define agent
        agent = StatelessAgent(net, nn.CrossEntropyLoss(), torch.optim.Adam(net.parameters()))

        # define world
        world = World(agent)

        # run world in training mode
        loss = world.train(train_loader, n_epochs=n_epochs)

        print loss

    def test_stateful_network(self):

        n_epochs = 30
        batch_size = 20

        # create toy data - predict if the sum of the input at the previous time step is larger than 1.0
        X = np.random.rand(1000, 2)
        T = (np.sum(X, 1) > 1.0).astype('int')
        T = np.hstack([0,T[1:]])
        train_data = TensorDataset(torch.Tensor(X), torch.CharTensor(T))

        # define data loader
        train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=False)

        # define network
        net = networks.RNN(n_input=2, n_output=2, n_hidden=3)

        # define agent
        agent = StatefulAgent(net, nn.CrossEntropyLoss(), torch.optim.Adam(net.parameters()), cutoff=5)

        # define world
        world = World(agent)

        # run world in training mode
        loss = world.train(train_loader, n_epochs=n_epochs)

        print loss


if __name__ == '__main__':
    unittest.main()