
import numpy as np

from collections import defaultdict

from chainer import Variable
import chainer.functions as F
from chainer import cuda


#####
## Basic REINFORCE agent
#

class REINFORCEAgent(object):
    """

    Implements REINFORCE algorithm

    https://webdocs.cs.ualberta.ca/%7Esutton/book/bookdraft2016sep.pdf
    https://github.com/dennybritz/reinforcement-learning/tree/master/PolicyGradient
    http://blog.shakirm.com/2015/11/machine-learning-trick-of-the-day-5-log-derivative-trick/
    http://www.1-4-5.net/~dmm/ml/log_derivative_trick.pdf
    """

    def __init__(self, model, optimizer=None, gamma=0.99, cutoff=None):
        """

        Args:
            model:
            optimizer:
            cutoff (int):
            gamma (0.99): Discounting factor
        """

        self.model = model

        self.optimizer = optimizer
        self.optimizer.setup(self.model)

        # discounting factor
        self.gamma = gamma

        self.cutoff = cutoff

        # monitor score and reward
        self.rewards = []
        self.scores = []

        # number of steps taken
        self.counter = 0

        # keep track of cumulative reward
        self.cum_reward = 0

    def train(self, observation, reward, done):
        """
        Trains agent on cumulate reward (return)
        
        Returns:
            action (Variable)
        """

        self.counter += 1

        # get reward associated with taking the previous action in the previous state
        if not reward is None:
            self.rewards.append(reward)

        # reset state since we started a new episode
        if done:
            self.reset_state()

        # compute policy and take new action based on observation, reward and terminal state
        action, policy = self.model(observation)

        # backpropagate
        if self.counter > 1 and (not self.model.has_state or self.counter % self.cutoff == 0 or done):

            # return value associated with last state
            R=0

            loss = 0
            for i in range(len(self.rewards)-1,-1,-1):

                R = self.rewards.pop() + self.gamma * R

                _ss = F.squeeze(self.scores.pop(),axis=1) * R

                if _ss.size > 1:
                    _ss = F.sum(_ss, axis=0)
                loss -= F.squeeze(_ss)

            self.optimizer.zero_grads()
            loss.backward()
            loss.unchain_backward()
            self.optimizer.update()

        # recompute score function: grad_theta log pi_theta (s_t, a_t) * v_t
        self.scores.append(self.score_function(action, policy))

        return cuda.to_cpu(action)

    def score_function(self, action, policy):
        """
        Computes score
        
        Args:
            action (int): 
            policy:

        Returns:
            score
        """

        logp = F.log_softmax(policy)

        score = F.select_item(logp, Variable(action))

        # handle case where we have only one element per batch
        if score.ndim == 1:
            score = F.expand_dims(score, axis=1)

        return score

    def test(self, observation, reward, done):
        """
        Tests agent

        Returns:
            action (Variable)
        """

        # reset state since we started a new episode
        if done:
            self.reset_state()

        # take new action based on observation, reward and terminal state
        action, _ = self.model(observation)

        return cuda.to_cpu(action)

    def reset_state(self):
        """
        Resets persistent states
        """

        if self.model.has_state:
            self.model.reset_state()
            self.counter = 0