'''implements tensor class that wraps numpy arrays.'''

import numpy as np

class Tensor(object):
    '''Tensor class holds a numpy array an pointers for autodiff graph.'''

    def __init__(self, data, parent=None):

        # pass arguments to numpy to create array structure.
        self.data = np.array(data)


        # some things may be a little easier if scalars are actually
        # rank 1 vectors.
        if self.data.shape == ():
            self.data = np.expand_dims(self.data, 0)

        # set up backward pointer for the computation graph
        self.parent = parent

        self.grad = None

    def detach(self):
        '''detach tensor from computation graph so that gradient computation stops.'''
        self.parent=None

    def backward(self, downstream_grad=None):
        if downstream_grad is None:
            downstream_grad = np.ones_like(self.data)

        if self.grad is None:
            self.grad = np.zeros_like(downstream_grad)
        self.grad += downstream_grad

        if self.parent is not None:
            self.parent.backward(downstream_grad)    



    