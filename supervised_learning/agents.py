from chainer import cuda

class SupervisedAgent(object):
    """
    Agent which trains on labelled data
    """

    def __init__(self, model, optimizer, cutoff=None):
        """
        Initializer for SupervisedAgent
        
        Args:
            model: 
            optimizer: 
            cutoff (int): 
        """

        self.model = model

        self.optimizer = optimizer
        self.optimizer.setup(self.model)

        # used for stateful networks only
        self.cutoff = cutoff
        self.counter = 0
        self.loss = None

    def __call__(self, data):
        """
        Runs networks in forward mode and applies optional output function
        
        Args:
            data
        
        Returns:
            post-processed output
        """

        return self.model.predict(data)

    def train(self, data):
        """
        Train agent on one batch
        :param data: 
        :return: loss
        """

        if self.model.predictor.has_state():

            self.counter += 1

            loss = self.model(data)

            if self.loss is None:
                self.loss = loss
            else:
                self.loss += loss

            if self.counter % self.cutoff == 0:

                self.optimizer.zero_grads()
                self.loss.backward()
                self.loss.unchain_backward()
                self.optimizer.update()

                self.loss = None

            return cuda.to_cpu(loss.data)

        else:

            self.optimizer.zero_grads()  # zero the gradient buffer
            loss = self.model(data)
            loss.backward()
            self.optimizer.update()
            return cuda.to_cpu(loss.data)

    def test(self, data):
        """
        Returns the loss for one batch
        
        Args:
            data
        
        Returns:
            loss
        """

        loss = self.model(data)

        return cuda.to_cpu(loss.data)

    def reset_state(self):
        """
        Resets persistent states
        """

        self.model.reset_state()
        self.counter = 0
