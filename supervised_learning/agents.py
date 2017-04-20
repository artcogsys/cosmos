from chainer import cuda

class Agent(object):

    def __init__(self, model, optimizer):

        self.model = model

        self.optimizer = optimizer
        self.optimizer.setup(self.model)

    def __call_(self, data):
        """
        Runs networks in forward mode and applies optional output function
        :param data:
        :return: post-processed output
        """

        return self.model.predict(data)

    def train(self, data):
        raise NotImplementedError

    def test(self, data):
        """
        Returns the loss for one batch
        :param data:
        :return: loss
        """

        loss = self.model(data)

        return cuda.to_cpu(loss.data)

    def reset_state(self):
        """
        Stateless agents don't require a state reset
        :return:
        """

        pass

class StatelessAgent(Agent):

    def train(self, data):
        """
        Trains on one batch and returns the loss
        :param data:
        :return: loss
        """

        self.optimizer.zero_grads()  # zero the gradient buffer
        loss = self.model(data)
        loss.backward()
        self.optimizer.update()

        return cuda.to_cpu(loss.data)

class StatefulAgent(Agent):

    def __init__(self, model, optimizer, cutoff):

        super(StatefulAgent, self).__init__(model, optimizer)

        self.cutoff = cutoff
        self.counter = 0

        self.loss = None

    def train(self, data):
        """
        Trains on one batch and returns the loss
        :param data: 
        :return: loss
        """

        self.counter += 1

        _loss = self.model(data)

        if self.loss is None:
            self.loss = _loss
        else:
            self.loss += _loss

        if self.counter % self.cutoff == 0:

            self.optimizer.zero_grads()
            self.loss.backward()
            self.loss.unchain_backward()
            self.optimizer.update()

            self.loss = None

        return cuda.to_cpu(_loss.data)

    def reset_state(self):
        self.model.reset_state()
        self.counter = 0
