import numpy as np

from chainer import Variable
import chainer.functions as F
from chainer import cuda


#####
## Basic REINFORCE agent
#

class REINFORCEAgent(object):
    """

    Implements REINFORCE algorithm

    """

    def __init__(self, model, optimizer=None, gamma=0.99, beta=1e-2, cutoff=None):
        """

        Args:
            model:
            optimizer:
            gamma (0.99): Discounting factor
            cutoff (int):
        """

        self.model = model

        self.optimizer = optimizer
        self.optimizer.setup(self.model)

        # discounting factor
        self.gamma = gamma

        # contribution of entropy term
        self.beta = beta

        self.cutoff = cutoff

        # monitor score and reward
        self.rewards = []
        self.scores = []
        self.entropies = []

        # number of steps taken
        self.counter = 0

        # keep track of cumulative reward
        self.cum_reward = 0

    def train(self, observation, reward, done):
        """
        Trains agent on cumulated reward (return)
        
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
        if len(self.rewards) > 0 and (not self.model.has_state
                                      or (not self.cutoff is None and self.counter % self.cutoff == 0)
                                      or done):

            # return value associated with last state
            R=0

            loss = 0
            for i in range(len(self.rewards)-1,-1,-1):

                R = self.rewards.pop() + self.gamma * R

                _ss = F.squeeze(self.scores.pop(),axis=1) * R

                if _ss.size > 1:
                    _ss = F.sum(_ss, axis=0)
                loss -= F.squeeze(_ss)

                loss -= self.beta * self.entropies.pop()

            self.optimizer.zero_grads()
            loss.backward()
            loss.unchain_backward()
            self.optimizer.update()

        # recompute score function: grad_theta log pi_theta (s_t, a_t) * v_t
        self.scores.append(self.score_function(action, policy))

        # compute entropy
        self.entropies.append(F.sum(self.entropy(policy)))

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

    def entropy(self, pi):
        """
        Computes entropy of policy

        Args:
            policy:
        """

        p = F.softmax(pi)
        logp = F.log_softmax(pi)

        return - F.sum(p * logp, axis=1)

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
        out = self.model(observation)

        return cuda.to_cpu(out[0])

    def reset_state(self):
        """
        Resets persistent states
        """

        if self.model.has_state:
            self.model.reset_state()
            self.counter = 0


#####
## Advantage Actor-Critic Agent
#

class AACAgent(REINFORCEAgent):
    """

    Implements Advantage Actor-Critic algorithm

    """

    def __init__(self, model, optimizer=None, gamma=0.99, beta=1e-2, cutoff=None):
        """

        Args:
            model:
            optimizer:
            gamma (0.99): Discounting factor
            beta (1e-2): weighting factor for entropy term (encourages exploration)
            cutoff (int):
        """

        super(AACAgent, self).__init__(model, optimizer=optimizer, gamma=gamma, beta=beta, cutoff=cutoff)

        self.values = []


    def train(self, observation, reward, done):
        """
        Trains agent on cumulated reward (return)

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

        # compute policy and value and take new action based on observation
        action, policy, value = self.model(observation)

        # backpropagate
        if len(self.rewards) > 0 and (not self.model.has_state
                                      or (not self.cutoff is None and self.counter % self.cutoff == 0)
                                      or done):

            # return value associated with last state
            if done:
                R=0
            else:
                R = value

            pi_loss = v_loss = 0
            for i in range(len(self.rewards) - 1, -1, -1):

                R = self.rewards.pop() + self.gamma * R

                advantage = R - self.values.pop()
                _ss = F.squeeze(self.scores.pop(), axis=1) * advantage.data

                if _ss.size > 1:
                    _ss = F.sum(_ss, axis=0)
                pi_loss -= F.squeeze(_ss)

                pi_loss -= self.beta * self.entropies.pop()

                v_loss += F.sum(advantage ** 2)

            # Compute total loss; 0.5 supposedly used by Mnih et al
            v_loss = F.reshape(v_loss, pi_loss.data.shape)
            loss = pi_loss + 0.5 * v_loss

            self.optimizer.zero_grads()
            loss.backward()
            loss.unchain_backward()
            self.optimizer.update()

        # recompute score function: grad_theta log pi_theta (s_t, a_t) * v_t
        self.scores.append(self.score_function(action, policy))

        # compute entropy
        self.entropies.append(F.sum(self.entropy(policy)))

        # add value
        self.values.append(value)

        return cuda.to_cpu(action)

class NESAgent(object):
    """

    Implements Natural Evolution Strategies algorithm

    https://blog.openai.com/evolution-strategies/
    https://gist.github.com/karpathy/77fbb6a8dac5395f1b73e7a89300318d
    
    hard to implement in same framework since the agent needs to run multiple versions of the task

    """

    def __init__(self, model, nsteps=100, npop=50, sigma=0.1, alpha=0.001):
        """

        Args:
            model:
            nsteps (100): number of steps to accumulate reward
            npop (50): population size
            sigma (0.1): noise standard deviation
            alpha (0.001): learning rate
        """

        self.model = model
        self.nsteps = nsteps
        self.npop = npop
        self.sigma = sigma
        self.alpha = alpha

        self.population = [self.model.copy() for i in range(self.npop)]

        self.rewards = np.zeros(self.npop)

        self.counter = 0

    def train(self, observation, reward, done):
        """
        Trains agent on cumulated reward (return)

        Returns:
            action (Variable)
        """

        if self.counter == 0:
            self.reset_population()

        action = self.predict(observation)

