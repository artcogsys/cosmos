import tqdm
import numpy as np
import copy

import chainer

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

    def train(self, train_iter, n_epochs, test_iter=None):

        train_loss = np.zeros([n_epochs, self.n_agents])

        validate = not test_iter is None
        if validate:
            min_loss = np.inf*np.ones(self.n_agents)
            test_loss = np.zeros([n_epochs, self.n_agents])/0
            _optimal_model = [None for i in range(self.n_agents)]

        for epoch in tqdm.tqdm(xrange(n_epochs)):

            map(lambda x: x.reset_state(), self.agents)

            with chainer.using_config('train', True):

                for data in train_iter:

                    train_loss[epoch] += map(lambda x: x.train(data), self.agents)

            # run validation
            if validate:

                test_loss[epoch] = self.test(test_iter)

                # store best model in case loss was minimized
                for i in range(self.n_agents):
                    if test_loss[epoch,i] < min_loss[i]:
                        _optimal_model[i] = copy.deepcopy(self.agents[i].model)
                        min_loss[i] = test_loss[epoch,i]

        # set models to optimal models based on test loss
        if validate:
            for i in range(self.n_agents):
                self.agents[i].model = copy.deepcopy(_optimal_model[i])

        return train_loss, test_loss

    def test(self, test_iter):

        test_loss =  np.zeros([1, self.n_agents])

        map(lambda x: x.reset_state(), self.agents)

        with chainer.using_config('train', False):

            for data in test_iter:
                test_loss += map(lambda x: x.test(data), self.agents)

        return test_loss