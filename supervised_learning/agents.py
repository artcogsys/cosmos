from torch.autograd import Variable
import torch

class StatelessAgent(object):

    def __init__(self, net, criterion, optimizer):

        self.net = net
        self.criterion = criterion
        self.optimizer = optimizer

    def train(self, data):

        self.optimizer.zero_grad()  # zero the gradient buffer
        outputs = self.net(Variable(data[0]))
        loss = self.criterion(outputs, Variable(data[1]))
        loss.backward()
        self.optimizer.step()

        return loss.data[0]

    def test(self, data):

        outputs = self.net(Variable(data[0]))
        loss = self.criterion(outputs, Variable(data[1]))

        return loss.data[0]

    def run(self, data):
        pass

    def reset(self):
        self.net.reset()

class StatefulAgent(object):

    def __init__(self, net, criterion, optimizer, cutoff):

        self.net = net
        self.criterion = criterion
        self.optimizer = optimizer

        # cutoff for BPTT
        self.cutoff = cutoff
        self.counter = 0

        self.loss = Variable(torch.zeros(1))

    def reset(self):
        self.net.reset()
        self.counter = 0

    def train(self, data):

        self.counter += 1

        outputs = self.net(Variable(data[0]))
        _loss = self.criterion(outputs, Variable(data[1]))
        self.loss += _loss

        if self.cutoff and (self.counter % self.cutoff) == 0:
            self.optimizer.zero_grad()  # zero the gradient buffer
            self.loss.backward(retain_variables=True)
            # self.net.h0.detach()
            # self.net.c0.detach()
            self.optimizer.step()

            self.loss = Variable(torch.zeros(1))

        return _loss.data[0]

    def test(self, data):

        outputs = self.net(Variable(data[0]))
        loss = self.criterion(outputs, Variable(data[1]))

        return loss.data[0]

    def run(self, data):
        pass

    def reset(self):
        self.net.reset()
