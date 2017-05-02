import numpy as np

import chainer
from chainer import Chain
import chainer.functions as F
from chainer import Variable

from chainer import cuda

#####
## Base class

class Model(Chain):
    """
    Model which wraps a network to generate predictions and compute policies
    """

    def __init__(self, net, gpu=-1):
        super(Model, self).__init__(predictor=net)

        if gpu >= 0:
            cuda.get_device(gpu).use()
            self.to_gpu()

    def __call__(self, data):
        raise NotImplementedError

    @property
    def has_state(self):
        """
        Checks if a network has persistent states

        Returns:
          bool
        """

        return self.predictor.has_state

    def reset_state(self):
        self.predictor.reset_state()


class ActorModel(Model):
    """
    An actor model computes the action and policy from a predictor
    """

    def __init__(self, net, gpu=-1):
        super(ActorModel, self).__init__(net, gpu=gpu)

    def __call__(self, data):
        """

        Args:
            data: observation
        
        Returns:
            action and policy
        """

        # linear outputs reflecting the log action probabilities and the value
        policy = self.predictor(Variable(self.xp.asarray(data)))

        # generate action according to policy
        p = F.softmax(policy).data

        # normalize p in case tiny floating precision problems occur
        row_sums = p.sum(axis=1)
        p /= row_sums[:, np.newaxis]

        action = self.xp.asarray([np.random.choice(p.shape[1], None, True, p[0])])

        return action, policy


class ActorCriticModel(Model):
    """
    An actor model computes the action and policy from a predictor
    """

    def __init__(self, net, gpu=-1):
        super(ActorCriticModel, self).__init__(net, gpu=gpu)

    def __call__(self, data):
        """

        Args:
            data: observation

        Returns:
            action and policy
        """

        out = self.predictor(Variable(self.xp.asarray(data)))

        # linear outputs reflecting the log action probabilities and the value
        policy = out[:, :-1]

        # final element is the value
        value = out[:, -1]

        # generate action according to policy
        p = F.softmax(policy).data

        # normalize p in case tiny floating precision problems occur
        row_sums = p.sum(axis=1)
        p /= row_sums[:, np.newaxis]

        action = self.xp.asarray([np.random.choice(p.shape[1], None, True, p[0])])

        return action, policy, value