import numpy as np

###
# Simple MDP task

class FooTask(object):
    """
    Very simple environment for testing fully observed models. The actor gets a reward when it correctly decides
    on the ground truth. Ground truth 0/1 determines probabilistically the number of 0s or 1s as observations

    """

    def __init__(self, n=2, p=0.8, batch_size=1, n_batches=np.inf):
        """

        :param n: number of inputs
        :param p: probability of emitting the right sensation at the input
        :param batch_size: one run by default
        :param n_batches: infinite number of time steps by default
        """

        if batch_size is None:
            batch_size = 1

        self.batch_size = batch_size
        self.n_batches = n_batches

        self.n_input = n
        self.p = p

        self.n_action = 1  # number of action variables
        self.n_output = 2  # number of output variables (actions) for the agent (discrete case)
        self.n_states = 1  # number of state variables

    def __iter__(self):

        self.idx = -1

        self.state = self.get_state()
        self.obs = self.get_observation()

        self.reward = None
        self.terminal = None

        return self

    def next(self):

        if self.idx == self.n_batches - 1:
            raise StopIteration

        self.idx += 1

        # return new observation and reward associated with agent's choice, we are always in a terminal state
        return self.obs, self.reward, np.ones(self.batch_size, dtype=np.bool)

    def process(self, agent):
        """ Process agent action, compute reward and generate new state and observation

        :param agent:
        :return:
        """

        self.reward = (2 * (agent.action == self.state) - 1).astype('float32')

        # this task always produces a new observation after each decision
        self.state = self.get_state()

        self.obs = self.get_observation()

    def get_state(self):
        """

        :return: new state
        """

        return np.random.choice(2, self.batch_size, True, [0.5, 0.5]).astype('int32')

    def get_observation(self):
        """

        :return: observation given the state
        """

        # produce a new observation at each step
        obs = np.zeros([self.batch_size, self.n_input]).astype('float32')
        for i in range(self.batch_size):

            if self.state[i] == 0:
                obs[i] = np.random.choice(2, [1, self.n_input], True, np.array([self.p, 1 - self.p]))
            else:
                obs[i] = np.random.choice(2, [1, self.n_input], True, np.array([1 - self.p, self.p]))

        return obs