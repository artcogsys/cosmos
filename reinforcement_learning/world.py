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

    def train(self, task, n_steps, snapshot=0):
        """
        
        Args:
            task: task to run agent(s) on
            n_steps (int): number of steps to train on
            snapshot (int): whether or not to save model after each epochs modulo snapshot
        
        Returns:
            rewards
        """

        rewards = np.zeros(n_steps)

        with chainer.using_config('train', True):

            map(lambda x: x.reset_state(), self.agents)

            obs, reward, done = task.reset()

            for step in tqdm.tqdm(xrange(n_steps)):

                # train step
                actions = map(lambda x: x.train(obs, reward, done), self.agents)

                # update step
                obs, reward, done = task.step(actions)

                rewards[step] = reward

                # store model every snapshot steps
                if snapshot and step % snapshot == 0:
                    for i in range(self.n_agents):
                        self.agents[i].model.save(os.path.join(self.out, 'agent-{0:04d}-epoch-{1:04d}'.format(i, step)))

        return rewards

    def test(self, task, n_steps):
        """
        
        Args:
            task: task to run agent(s) on
            n_steps (int): number of steps to train on
       
        Returns:
            test loss and reward
        """

        rewards = np.zeros(n_steps)

        with chainer.using_config('train', False):

            map(lambda x: x.reset_state(), self.agents)

            obs, reward, done = task.reset()

            for step in tqdm.tqdm(xrange(n_steps)):

                # test step
                actions = map(lambda x: x.test(obs, reward, done), self.agents)

                # update step
                obs, reward, done = task.step(actions)

                rewards[step] = reward

        return rewards