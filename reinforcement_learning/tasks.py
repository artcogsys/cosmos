import numpy as np

class EvidenceTask(object):
    """
    Very simple task which only requires evaluating present evidence and does not require evidence integration. 
    The actor gets a reward when it correctly decides on the ground truth. Ground truth 0/1 determines probabilistically 
    the number of 0s or 1s as observations
    """

    def __init__(self, n=2, p=0.8):
        """
        
        Args:
            n: number of inputs (pieces of evidence)
            p: probability of emitting the right sensation at the input
        """

        self.n_input = n
        self.p = p

        self._state = None

        # self.n_action = 1  # number of action variables
        # self.n_output = 2  # number of output variables (actions) for the agent (discrete case)
        # self.n_states = 1  # number of state variables

    def reset(self):
        """
        Resets state and generates new observations
        
        Returns:
            observations, reward, done
        """

        # generate state
        self._state = np.random.choice(2, 1, True, [0.5, 0.5]).astype('int32')

        # generate associated observations
        P = [self.p, 1 - self.p] if self._state ==0 else [1 - self.p, self.p]
        obs = np.random.choice(2, self.n_input, True, P).astype('float32').reshape([1, self.n_input])

        return obs, None, True

    def step(self, action):
        """
        This task always produces a new state and observation after each decision
        
        Args:
            action: agent(s) action
        :return: 
        """

        # handling of multiple agents by only assigning reward if all agents make a correct prediction
        reward = (2 * np.all(action == self._state) - 1).astype('float32')

        # generate state
        self._state = np.random.choice(2, 1, True, [0.5, 0.5]).astype('int32')

        # generate associated observations
        P = [self.p, 1 - self.p] if self._state == 0 else [1 - self.p, self.p]
        obs = np.random.choice(2, self.n_input, True, P).astype('float32').reshape([1, self.n_input])

        return obs, reward, True