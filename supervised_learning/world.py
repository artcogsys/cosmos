import tqdm
import numpy as np

class World(object):

    def __init__(self, agents):
        """ A world is inhabited by one or more agents

        :param agents:
        """

        if not isinstance(agents, list):
            self.agents = [agents]
        else:
            self.agents = agents

        self.n_agents = len(self.agents)

    def train(self, train_loader, n_epochs):

        losses = np.zeros([n_epochs, self.n_agents])

        for epoch in tqdm.tqdm(xrange(n_epochs)):

            map(lambda x: x.reset(), self.agents)

            for i, data in enumerate(train_loader):

                losses[epoch] += map(lambda x: x.train(data), self.agents)

        return losses