import tqdm
import numpy as np
import copy
import os

import chainer

class World(object):
    """
    Wrapper object which takes care of training and testing on some data iterator for one or more agents
    """

    def __init__(self, agents, out='result'):
        """ A world is inhabited by one or more agents

        Args:
            agents: Agents that are run on this world
            out: output folder
        """

        if not isinstance(agents, list):
            self.agents = [agents]
        else:
            self.agents = agents

        self.n_agents = len(self.agents)

        self.out = out
        if not self.out is None:
            try:
                os.makedirs(self.out)
            except OSError:
                pass

    def train(self, train_iter, n_epochs, test_iter=None, snapshot=0):
        """
        
        Args:
            train_iter: iterator over the training data
            n_epochs (int): number of epochs to train on
            test_iter: optional iterator over the test data (returns optimal model)
            snapshot (int): whether or not to save model after each epochs modulo snapshot
        
        Returns:
            train loss and optional test loss
        """

        train_loss = np.zeros([n_epochs, self.n_agents])

        validate = not test_iter is None
        if validate:
            min_loss = np.inf*np.ones(self.n_agents)
            test_loss = np.zeros([n_epochs, self.n_agents])/0
            _optimal_model = [None for i in range(self.n_agents)]

        for epoch in tqdm.tqdm(xrange(n_epochs)):

            map(lambda x: x.reset_state(), self.agents)

            # train step
            with chainer.using_config('train', True):

                for data in train_iter:

                    train_loss[epoch] += map(lambda x: x.train(data), self.agents)

            # store model every snapshot epochs
            if snapshot and epoch % snapshot == 0:
                for i in range(self.n_agents):
                    self.agents[i].model.save(os.path.join(self.out, 'agent-{0:04d}-epoch-{1:04d}'.format(i, epoch)))

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
        """
        
        Args:
            test_iter: iterator over the test data
        
        Returns:
            test loss
        """

        test_loss =  np.zeros([1, self.n_agents])

        map(lambda x: x.reset_state(), self.agents)

        with chainer.using_config('train', False):

            for data in test_iter:
                test_loss += map(lambda x: x.test(data), self.agents)

        return test_loss