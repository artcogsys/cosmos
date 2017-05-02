import unittest

import numpy as np
import matplotlib.pyplot as plt

import reinforcement_learning.networks as networks
from reinforcement_learning.agents import *
from reinforcement_learning.models import *
from reinforcement_learning.world import World
from reinforcement_learning.tasks import EvidenceTask

n_steps = 1000

task = EvidenceTask()

# REINFORCE

# define network
net = networks.RNN(n_output=2, n_hidden=10)

# define agent
agent = REINFORCEAgent(ActorModel(net, gpu=-1), chainer.optimizers.Adam())

# define world
world = World(agent)

# run world in training mode
rewards1 = np.cumsum(world.train(task, n_steps=n_steps))

# AAC

# define network
net = networks.RNN(n_output=3, n_hidden=10)

# define agent
agent = AACAgent(ActorCriticModel(net, gpu=-1), chainer.optimizers.Adam(), beta=0)

# define world
world = World(agent)

# run world in training mode
rewards2 = np.cumsum(world.train(task, n_steps=n_steps))

plt.plot(np.vstack([rewards1, rewards2]).T)
plt.legend(['REINFORCE', 'AAC'])
plt.show()
