from chainer import Chain
import chainer.links as L
import chainer.functions as F

class MLP(Chain):

    def __init__(self, n_input=None, n_output=1, n_hidden=10):
        super(MLP, self).__init__(l1=L.Linear(n_input, n_hidden), l2=L.Linear(n_hidden, n_output))

    def __call__(self, x):
        return self.l2(F.relu(self.l1(x)))

    def reset_state(self):
        pass

class RNN(Chain):

    def __init__(self, n_input=None, n_output=1, n_hidden=10):
        super(RNN, self).__init__(lstm=L.LSTM(n_input, n_hidden), fc=L.Linear(n_hidden, n_output))

    def __call__(self, x):

        return self.fc(self.lstm(x))

    def reset_state(self):
        self.lstm.reset_state()
