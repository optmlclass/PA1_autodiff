'''SGD implementation'''

from variable import Variable



def SGD(loss_fn, params, data, learning_rate):
    ''''performs an SGD update and returns updated parameters.
    arguments:
        loss_fn: function that takes params, data and returns a loss value.
        params: list of Variables representing parameters of some model.
        data: list of Variables representing minibatch data.
        learning_rate: learning rate for SGD.
    returns: two values:
        new_params: Variable containing next value for params after SGD update,
        correct: float number that is 1.0 if the prediction was correct, and 
            0.0 otherwise.
    '''

    ### YOUR CODE HERE ###

    raise NotImplementedError


    # Depending on your solution, you may need to uncomment the lines below for
    # it to run efficiently. If your solution requires this, make sure you
    # understand why!
    # for param in new_params:
    #     param.detach()

    return new_params, correct







