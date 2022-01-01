import numpy as np
import functools
from operation import Operation
from tensor import Tensor


class TensorDot(Operation):
    def __init__(self):
        super(TensorDot, self).__init__(name="TensorDot")

    def forward_call(self, A, B, dims_to_contract):
        '''A and B are are Variables, dims_to_contract is number
        of dimensions to contract. This is a special case of np.tensordot.
        Example:
        A is dim [2, 3, 4]
        B is dim [3, 4, 5]

        if dims_to_contract is 2, output will be [2, 5]
        Otherwise it is an error.
        '''

        self.parents = [A, B]
        self.dims_to_contract = dims_to_contract

        return np.tensordot(A.data, B.data, dims_to_contract)

    def backward_call(self, downstream_grad):
        A = self.parents[0]
        B = self.parents[1]
        A_indices = np.arange(0, len(A.data.shape) - self.dims_to_contract)
        B_indices = np.arange(self.dims_to_contract, len(B.data.shape))
        A_grad = np.tensordot(downstream_grad, B.data, [B_indices, B_indices])
        B_grad = np.tensordot(A.data, downstream_grad, [A_indices, A_indices])
        return [A_grad, B_grad]


class TensorAdd(Operation):
    '''tensor addition operation.'''

    def __init__(self):
        super(TensorAdd, self).__init__(name="Add")

    def forward_call(self, summands):
        '''summands should be a list of ndarrays, all with the same dimensions.'''

        assert len(summands) > 0, "Add called with no inputs!"
        shape = summands[0].data.shape
        for summand in summands:
            assert summand.data.shape == shape, "Shape mismatch in Add: shapes: {}".format(
                [summand.data.shape for summand in summands])

        self.parents = summands

        return functools.reduce(lambda x, y: x.data + y.data, summands)

    def backward_call(self, downstream_grad):
        return [downstream_grad for _ in self.parents]

class TensorMultiply(Operation):
    '''coordinate-wise multiply operation.'''

    def __init__(self):
        super(TensorMultiply, self).__init__(name="Multiply")

    def forward_call(self, multiplicands):
        '''inputs should be a list of ndarrays, all with the same dimensions.'''

        assert len(multiplicands) > 0, "Multiply called with no inputs!"
        shape = multiplicands[0].data.shape
        for multiplicand in multiplicands:
            assert multiplicand.data.shape == shape, "Shape mismatch in Multiply!"

        self.num_inputs = len(multiplicands)
        self.output = functools.reduce(
            lambda x, y: x.data*y.data, multiplicands)

        self.parents = multiplicands

        return self.output

    def backward_call(self, downstream_grad):

        # gradient of abcd with respect to b is acd
        # cannot just do abcd/b because of potential divide by zero.
        upstream_grads = []
        for i in range(len(self.parents)):
            product = downstream_grad
            for j, multiplicand in enumerate(self.parents):
                if i != j:
                    product = product * multiplicand.data
            upstream_grads.append(product)

        return upstream_grads

class Power(Operation):
    '''raise to a power'''

    def __init__(self, exponent):
        super(Power, self).__init__(name="{} Power".format(exponent))

        self.exponent = exponent

    def forward_call(self, tensor):
        self.parents = [tensor]

        return np.power(tensor.data, self.exponent)

    def backward_call(self, downstream_grad):
        tensor = self.parents[0]
        return [downstream_grad * self.exponent * np.power(tensor.data, self.exponent - 1.0)]


class Exp(Operation):
    '''exponentiate'''

    def __init__(self):
        super(Exp, self).__init__(name="exp")

    def forward_call(self, tensor):
        self.parents = [tensor]
        self.output = np.exp(tensor.data)
        return self.output

    def backward_call(self, downstream_grad):
        return [downstream_grad * self.output]


class Maximum(Operation):
    '''computes coordinate-wise maximum of a list of tensors'''

    def __init__(self):
        super(Maximum, self).__init__(name="maximum")

    def forward_call(self, terms):
        self.parents = terms
        self.output = functools.reduce(
            lambda x, y: np.maximum(x, y), [t.data for t in terms])

        return self.output

    def backward_call(self, downstream_grad):
        masks = [t.data == self.output for t in self.parents]

        return [m * downstream_grad for m in masks]

class ReLU(Operation):
    '''computes coordinate-wise maximum with 0'''

    def __init__(self):
        super(ReLU, self).__init__(name="relu")

    def forward_call(self, A):
        self.parents = [A]
        self.output = np.maximum(A.data, 0.0)

        return self.output

    def backward_call(self, downstream_grad):
        mask = (self.output == self.parents[0].data)
        return [mask * downstream_grad]

class ReduceMax(Operation):
    '''computes the maximum element of a tensor'''

    def __init__(self):
        super(ReduceMax, self).__init__(name="ReduceMax")

    def forward_call(self, A):
        self.parents = [A]
        self.output = np.max(A.data)

        return self.output

    def backward_call(self, downstream_grad):
        A = self.parents[0]

        mask = (A.data == self.output)
        return [mask * downstream_grad]


class ScalarMultiply(Operation):
    '''multiplication by a scalar.'''

    def __init__(self):
        super(ScalarMultiply, self).__init__(name="Scalar Multiply")

    def forward_call(self, scalar, tensor):

        assert scalar.data.size == 1, "ScalarMultiply called with non-scalar input!"

        self.parents = [scalar, tensor]

        return scalar.data * tensor.data

    def backward_call(self, downstream_grad):

        [scalar, tensor] = self.parents

        # gradient of abcd with respect to b is acd
        upstream_grads = [
            np.sum(tensor.data * downstream_grad),
            downstream_grad * scalar.data
        ]

        return upstream_grads


class HingeLoss(Operation):
    '''compute hinge loss.
    assumes input are a scores for a single example (no need to support
    minibatches of scores).
    
    Input "scores" will be [1 x C] tensor representing scores for each of
    C classes.
    "label" is an integer in [0,..., C-1].
    The multi-class hinge loss is given by the (unweighted) formula here:
    https://pytorch.org/docs/stable/generated/torch.nn.MultiMarginLoss.html
    
    '''

    def __init__(self, label):
        super(HingeLoss, self).__init__(name="Hinge Loss")
        self.label = label

    def forward_call(self, scores):
        
        self.parents = [scores]
        value = np.sum(np.maximum(scores.data - scores.data[self.label] + 1.0, 0.0))/len(scores.data)
        self.value = value
        return value

    def backward_call(self, downstream_grad):
        [scores] = self.parents

        over_margins = np.maximum(scores.data - scores.data[self.label] + 1.0, 0.0) > 0.0

        upstream_grad = np.zeros_like(scores.data)

        upstream_grad[over_margins] = 1
        upstream_grad[self.label] = -1 * (np.sum(over_margins) - 1.0)
        upstream_grad /= len(over_margins)

        return [upstream_grad]


class MatrixMultiply(Operation):
    def __init__(self):
        super(MatrixMultiply, self).__init__(name="MatrixMultiply")

    def forward_call(self, A, B):
        self.parents = [A, B]

        return np.dot(A.data, B.data)

    def backward_call(self, downstream_grad):
        A = self.parents[0]
        B = self.parents[1]
        A_grad = np.dot(downstream_grad, np.transpose(B.data))
        B_grad = np.dot(np.transpose(A.data), downstream_grad)
        return [A_grad, B_grad]



###### Helper functions for operator overloading ######

Tensor.__add__ = lambda self, other: TensorAdd()([self, other])

def mul(self, other):
    if not isinstance(other, Tensor):
        other = Tensor(other)
    if other.data.size == 1:
        return ScalarMultiply()(other, self)
    else:
        return TensorMultiply()([self, other])
Tensor.__mul__ = Tensor.__rmul__ = mul

Tensor.__neg__ = lambda self: -1 * self
Tensor.__sub__ = lambda self, other: self + (- other)

def div(A, B):
    if not isinstance(B, Tensor):
        B = Tensor(B)
    B_inverse = Power(-1.0)(B)
    if B.data.size == 1:
        return ScalarMultiply()(B_inverse, A)
    else:
        return TensorMultiply()([A, B_inverse])

Tensor.__truediv__ = div
Tensor.__rtruediv__ = lambda self, other: div(other, self)


###### Helper functions for applying operations ######

def matmul(a, b):
    mm = MatrixMultiply()
    return mm(a, b)

def pow(tensor, exponent):
    power = Power(exponent)
    return power(tensor)

def tensordot(A, B, dims_to_contract):
    tensordot_op = TensorDot()
    return tensordot_op(A, B, dims_to_contract)

def relu(A):
    relu_op = ReLU()
    return relu_op(A)

