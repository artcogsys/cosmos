from chainer import Chain
import chainer.links as L
import chainer.functions as F

class MLP(Chain):
    """
    Multilayer perceptron
    """

    def __init__(self, n_input=None, n_output=1, n_hidden=10):
        super(MLP, self).__init__(l1=L.Linear(n_input, n_hidden), l2=L.Linear(n_hidden, n_output))

    def __call__(self, x):
        return self.l2(F.relu(self.l1(x)))

    def has_state(self):
        """
        Checks if a network has persistent states
        
        Returns:
           bool
        """

        return False

    def reset_state(self):
        """
        Resets persistent states
        
        """

        pass

class RNN(Chain):

    def __init__(self, n_input=None, n_output=1, n_hidden=10):
        super(RNN, self).__init__(lstm=L.LSTM(n_input, n_hidden), fc=L.Linear(n_hidden, n_output))

    def __call__(self, x):
        return self.fc(self.lstm(x))

    def has_state(self):
        """
        Checks if a network has persistent states
        
        Returns:
           bool
        """

        return True

    def reset_state(self):
        """
        Resets persistent states
        """

        self.lstm.reset_state()
