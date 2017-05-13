from collections import defaultdict
import copy
import numpy as np

#####
## Monitor base class

class Monitor(object):

    def __init__(self, variables, keys=None):
        """

        :param variables: reference copy of list of variables that must be stored
        :param keys: if defined then the list indicates keys to store
        """

        self.dict = defaultdict(list)

        self.variables = variables

        if keys is None:
            self.keys = range(len(variables))
        else:
            self.keys = keys

    def store(self):
        for i in range(len(self.keys)):
            self.dict[self.keys[i]].append(self.variables[i])

    def __getitem__(self, item):
        """Returns dict[item]

        :param item: key name
        :return: dictionary item
        """

        assert(item in self.keys)
        return self.dict[item]