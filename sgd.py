'''SGD implementation'''
from .tensor import Tensor
import numpy as np



def SGD(loss_fn, params, data, learning_rate):
    ''''performs an SGD update and returns updated parameters.
    arguments:
        loss_fn: function that takes params, data and returns a loss value.
        params: list of Tensors representing parameters of some model.
        data: list of Tensors representing minibatch data.
        learning_rate: learning rate for SGD.
    returns:
        Tensor containing next value for params after SGD update.
    '''

    ### YOUR CODE HERE ###

    loss_val, correct = loss_fn(params, data)
    loss_val.backward()

    new_params = [param - Tensor(learning_rate * param.grad) for param  in params]
    
    # this part is very important! without it you will quickly run out of memory because
    # the backward will backprop through the entire training run on every iteration!
    # Alternatively, you could define the new_params via numpy operations that
    # don't record gradients, as in:
    # new_params = [Tensor(param.data - learning_rate * param.grad) for param  in params]
    # and then it would not be necessary to include the detach below.
    for param in new_params:
        param.detach()

    return new_params, correct







