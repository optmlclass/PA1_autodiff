'''implements tensor class that wraps numpy arrays.'''

import numpy as np

class Tensor(object):
    '''Tensor class holds a numpy array an pointers for autodiff graph.'''

    def __init__(self, data, parent=None):

        # pass arguments to numpy to create array structure.
        self.data = np.array(data)

        # some things may become a little easier if we convert scalars into
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
        '''
        backward pass.
        args:
            downstream_grad:
                numpy array representing derivative of final output
                with respect to this tensor.
        
        This function returns no values, but accomplishes two tasks:
        1. accumulate the downstream_grad in the self.grad attribute so that
        at the end of all backward passes, the self.grad attribute contains
        the gradient of the final output with respect to this tensor.

        Note that this is NOT accomplished by self.grad = downstream_grad!

        2. pass downstream_grad to the parent operations that created this
        Tensor so that the backpropogation can continue.
        '''
        # set a default value for downstream_grad.
        # if the backward is called  on the output tensor and the output
        # is a scalar, this will result in the standard gradient calculuation.
        if downstream_grad is None:
            downstream_grad = np.ones_like(self.data)

        if self.grad is None:
            self.grad = np.zeros_like(downstream_grad)


        ### YOUR CODE HERE ###
        self.grad += downstream_grad

        if self.parent is not None:
            self.parent.backward(downstream_grad)    



    