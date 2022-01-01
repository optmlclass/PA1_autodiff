
import numpy as np

from tensor import Tensor

class Operation(object):
    '''Base class for operations'''

    def __init__(self, name):

        # list of input tensors to this operation, and output tensor.
        # All inputs and outputs be Tensor objects.
        self.parents = None
        self.child = None

        # flag for error-checking
        self.name = name
        self.forward_called = False


    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        '''wrapper around forward_call to set flag.'''
        self.forward_called = True
        output = self.forward_call(*args, **kwargs)
        assert self.parents is not None, "forward did not set self.parents on {} operation! Inputs: args: {}, kwargs: {}. self.parents: {}".format(
            self.name, args, kwargs, self.parents)
        return Tensor(data=output, parent=self)

    def backward(self, downstream_grad):
        '''wrapper around backward_call to check assertion.'''

        assert self.forward_called, "backward called before forward on {} operation!".format(
            self.name)
        upstream_grads = self.backward_call(downstream_grad)
        for var, grad in zip(self.parents, upstream_grads):
            var.backward(grad)

    def backward_call(self, downstream_grad):
        '''Performs backward pass.

        This function should also set self.gradients in such a way that
        self.gradients[i] is the gradient of the final output of the computation
        graph with respect to self.inputs[i].

        Args:
            downstream_grad: gradient from downstream operation in the
                computation graph. This package will only consider
                computation graphs that result in scalar outputs at the final
                node (e.g. loss function computations). As a result,
                the dimension of downstream_grad should match the dimension of the
                output of this operation class.

                Formally, if this operation computes F(x), and the final
                computation computes a scalar, G(F(x)), then downstream_grad is
                dG/dF.

                If F(x)\in R^n, then downstream_grad should be a map from R^n -> R
                and so is a 1 x n tensor. If F(x) \in R^(a x b) (i.e. a matrix)
                then downstream_grad is a tensor represnting a map R^(a x b) -> R
                an so is a 1 x a x b tensor.

                You can choose to drop the extra "1" dimension at the front
                if you desire.
        returns
            list of gradients to pass to upstream operations. The size of this
                list equals the number of inputs to the operation.

                Example:
                If there are N inputs, and the output is F(x_1,...,x_N), then
                the ith element of this list is equal to
                downstream_grad * dF/dx_i(x_1,..,x_n), where * indicates
                a tensor contraction.
                
                In the simplest case, dF/dx_i is a matrix of dimension
                m * n where n is the dimension of x_i and m is the dimension of
                the output F (so that dF/dx_i is a linear  map R^n -> R^m)
                In this case, downstream_grad, is a 1 x m vector so that
                downstream_grad * dF/dx_i(x_1,..,x_n) has dimension 1 x n,
                which is the appropriate dimension for dG/dx_i where G is the
                final output in R.
        '''
        raise NotImplementedError

        def forward_call(self, *args, **kwargs):
            '''forward pass. Should compute operation and save relevant state
            needed for backward pass.
            Args:
                inputs: inputs to this operation.
            returns output of operation as a numpy array
            '''
            raise NotImplementedError
