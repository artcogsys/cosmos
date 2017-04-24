import chainer
from chainer import Chain
import chainer.functions as F
from chainer import Variable

from chainer import cuda

#####
## Base class

class Model(Chain):
    """
    Model which wraps a network to compute loss and generate actual predictions
    """

    def __init__(self, net, loss_function, output_function=lambda x:x, gpu=-1):
        super(Model, self).__init__(predictor=net)

        self.loss_function = loss_function
        self.output_function = output_function

        if gpu >= 0:
            cuda.get_device(gpu).use()
            self.to_gpu()

    def __call__(self, data):
        """ Compute loss for minibatch of data

        Args:
            data: list of minibatches (e.g. inputs and targets for supervised learning)
        
        Returns:
            loss
        """

        data = map(lambda x: Variable(self.xp.asarray(x)), data)

        # handle settings where we have more than one input dataset
        x = data[0] if len(data) == 2 else data[:-1]  # inputs
        target = data[-1]  # targets

        loss = self.loss_function(self.predictor(x), target)

        return loss

    def predict(self, data):
        """
        Returns prediction, which can be different than raw output (e.g. for softmax function)

        Args:
            data: minibatch or list of minibatches representing input to the model
        
        Returns:
            prediction
        """

        data = map(lambda x: Variable(self.xp.asarray(x)), data)

        return self.output_function(self.predictor(data)).data

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

#####
## Classifier object

class Classifier(Model):
    """
    Wrapper for classification problems
    """

    def __init__(self, net, gpu=-1):
        super(Classifier, self).__init__(net, gpu=gpu, loss_function=F.softmax_cross_entropy,
                                         output_function=F.softmax)

#####
## Regressor object

class Regressor(Model):
    """
    Wrapper for regression problems
    """

    def __init__(self, net, gpu=-1):
        super(Regressor, self).__init__(net, gpu=gpu, loss_function=F.mean_squared_error)