import numpy as np

class RandomIterator(object):

    def __init__(self, data, batch_size=None):
        """
        
        :param data: 
        :param batch_size:
        :return list of batches
        """

        self.data = data

        if batch_size is None:
            batch_size = 1

        self.batch_size = batch_size
        self.n_batches = len(self.data) // batch_size

    def __iter__(self):

        self.idx = -1
        self._order = np.random.permutation(len(self.data))[:(self.n_batches * self.batch_size)]

        return self

    def next(self):

        self.idx += 1

        if self.idx == self.n_batches:
            raise StopIteration

        i = self.idx * self.batch_size

        return list(self.data[self._order[i:(i + self.batch_size)]])


class SequentialIterator(object):

    def __init__(self, data, batch_size=None):

        self.data = data

        self.batch_size = batch_size
        self.n_batches = len(self.data) // batch_size

    def __iter__(self):

        self.idx = -1

        offsets = [i * self.n_batches for i in range(self.batch_size)]

        # define custom ordering; we won't process beyond the end of the trial
        self._order = []
        for iter in range(self.n_batches):
            x = [(offset + iter) % len(self.data) for offset in offsets]
            self._order += x

        return self

    def next(self):

        self.idx += 1

        if self.idx == self.n_batches:
            raise StopIteration

        i = self.idx * self.batch_size

        return list(self.data[self._order[i:(i + self.batch_size)]])