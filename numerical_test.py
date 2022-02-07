import functools
import numpy as np
import unittest

import ops_impl as ops
from variable import Variable


def numerical_grad(inputs, downstream_grad, evaluate):
    '''computes numerical gradients.
    Args:
        inputs: list of np.arrays
        downstream_grad: np.array
        evaluate: taks a list of input arrays and produces an output array
            of same shape as downstream_grad
    returns a list of np.arrays such that the ith element of the return value
        is the gradient of np.sum(evaluate(inputs) * downstream_grad)
        with respect to the ith element of inputs.
    '''
    delta = 1e-8

    base_function_value = np.sum(downstream_grad * evaluate(inputs))

    gradients = []

    for i in range(len(inputs)):
        grad = np.zeros(inputs[i].size)
        perturbation = np.zeros(inputs[i].size)
        for j in range(inputs[i].size):
            perturbation[j] = delta
            inputs[i] = inputs[i] + np.reshape(perturbation, inputs[i].shape)
            perturbed_value = np.sum(downstream_grad * evaluate(inputs))
            inputs[i] = inputs[i] - np.reshape(perturbation, inputs[i].shape)
            perturbation[j] = 0.0
            grad[j] = (perturbed_value - base_function_value) / delta
        gradients.append(np.reshape(grad, inputs[i].shape))

    return gradients


def test_backward_random(input_shapes, output_shape, reference_fn, operation_fn, positive=False):
    args = [np.random.normal(size=shape) for shape in input_shapes]
    if positive:
        args = [np.abs(arg) for arg in args]
    downstream_grad = np.ones(output_shape)

    numeric = numerical_grad(args, downstream_grad, reference_fn)

    tensors = [Variable(arg) for arg in args]
    output = operation_fn(tensors)
    output.backward(downstream_grad)

    analytic = [var.grad.data for var in tensors]
    diff = np.sum([np.linalg.norm(a-n)/(1e-10 + np.linalg.norm(a+n))
                   for a, n in zip(numeric, analytic)])
    return diff


def test_forward_random(input_shapes, reference_fn, operation_fn, positive=False):
    args = [np.random.normal(size=shape) for shape in input_shapes]
    if positive:
        args = [np.abs(arg) for arg in args]
    tensors = [Variable(arg) for arg in args]
    analytic = operation_fn(tensors).data

    reference = reference_fn(args)

    diff = np.linalg.norm(analytic-reference) / \
        (1e-10 + np.linalg.norm(analytic+reference))
    return diff

class TestAutograd(unittest.TestCase):

    def test_add(self):
        input_shapes = [(2, 3), (2, 3), (2, 3)]
        output_shape = (2, 3)

        def reference_fn(args):
            return sum(args)

        def operation_fn(args):
            add = ops.VariableAdd()
            return add(args)
        self._test_op(input_shapes, output_shape, reference_fn,
                      operation_fn, positive=False)


    def test_overload_add(self):
        input_shapes = [(2, 3), (2, 3), (2, 3)]
        output_shape = (2, 3)

        def reference_fn(args):
            return sum(args)

        def operation_fn(args):
            return functools.reduce(lambda x, y: x + y, args)
        self._test_op(input_shapes, output_shape, reference_fn,
                      operation_fn, positive=False)

    def test_mul(self):
        input_shapes = [(2, 3), (2, 3), (2, 3)]
        output_shape = (2, 3)

        def reference_fn(args):
            return functools.reduce(lambda x, y: x*y, args)

        def operation_fn(args):
            mul = ops.VariableMultiply()
            return mul(args)

        self._test_op(input_shapes, output_shape, reference_fn,
                      operation_fn, positive=False)

    def test_mul_by_zero(self):

        inputs = [Variable([10.0, 0.0, 4.0]), Variable([0.0, 2.0, 2.0])]
        
        mul = ops.VariableMultiply()

        output = mul(inputs)

        expected_output = np.array([0.0, 0.0, 8.0])
        expected_grads = [np.array([0.0, 2.0, 2.0]), np.array([10.0, 0.0, 4.0])]

        def relative_error(a, b):
            diff = np.linalg.norm(a - b) / (1e-10 + np.linalg.norm(a + b))
            return diff
            
        self.assertLessEqual(relative_error(output.data, expected_output), 1e-5)

        output.backward()

        self.assertLessEqual(relative_error(inputs[0].grad, expected_grads[0]), 1e-5)
        self.assertLessEqual(relative_error(inputs[1].grad, expected_grads[1]), 1e-5)

    def test_overload_mul(self):
        input_shapes = [(2, 3), (2, 3), (2, 3)]
        output_shape = (2, 3)

        def reference_fn(args):
            return functools.reduce(lambda x, y: x*y, args)

        def operation_fn(args):
            return functools.reduce(lambda x, y: x*y, args)

        self._test_op(input_shapes, output_shape, reference_fn,
                      operation_fn, positive=False)

    def test_overload_sub(self):
        input_shapes = [(2, 3), (2, 3), (2, 3)]
        output_shape = (2, 3)

        def reference_fn(args):
            return functools.reduce(lambda x, y: x - y, args)

        def operation_fn(args):
            return functools.reduce(lambda x, y: x - y, args)

        self._test_op(input_shapes, output_shape, reference_fn,
                      operation_fn, positive=False)


    def test_overload_div(self):
        input_shapes = [(2, 3), (2, 3), (2, 3)]
        output_shape = (2, 3)

        def reference_fn(args):
            return functools.reduce(lambda x, y: x / y, args)

        def operation_fn(args):
            return functools.reduce(lambda x, y: x / y, args)

        self._test_op(input_shapes, output_shape, reference_fn,
                      operation_fn, positive=False)

    def test_scalar_multiply(self):
        input_shapes = [(1), (2, 3)]
        output_shape = (2, 3)

        def reference_fn(args):
            return args[0]*args[1]

        def operation_fn(args):
            mul = ops.ScalarMultiply()
            return mul(*args)
        self._test_op(input_shapes, output_shape, reference_fn,
                      operation_fn, positive=False)

    def test_add_uses_downstream(self):
        input_shapes = [(2, 3), (2, 3), (2, 3)]
        output_shape = (2, 3)

        scaleFactor = 10.0

        def reference_fn(args):
            return scaleFactor * sum(args)

        def operation_fn(args):
            add = ops.VariableAdd()
            
            mul = ops.ScalarMultiply()
            scaleFactorVariable = Variable(scaleFactor)
            return mul(scaleFactorVariable, add(args))
        self._test_op(input_shapes, output_shape, reference_fn,
                      operation_fn, positive=False)

    def test_matrix_multiply(self):
        input_shapes = [(4, 2), (2, 3)]
        output_shape = (4, 3)

        def reference_fn(args):
            return np.dot(args[0], args[1])

        def operation_fn(args):
            mul = ops.MatrixMultiply()
            return mul(*args)
        self._test_op(input_shapes, output_shape, reference_fn,
                      operation_fn, positive=False)

    def test_tensordot(self):
        input_shapes = [(2, 4, 2, 6), (2, 6, 3)]
        output_shape = (2, 4, 3)

        def reference_fn(args):
            return np.tensordot(args[0], args[1], 2)

        def operation_fn(args):
            dot = ops.TensorDot()
            return dot(*args, dims_to_contract=2)
        self._test_op(input_shapes, output_shape, reference_fn,
                      operation_fn, positive=False)

    def test_exponent(self):
        input_shapes = [(2, 3, 4)]
        output_shape = (2, 3, 4)

        def reference_fn(args):
            return np.exp(args[0])

        def operation_fn(args):
            exp = ops.Exp()
            return exp(args[0])

        self._test_op(input_shapes, output_shape, reference_fn,
                      operation_fn, positive=False)

    def test_power(self):
        input_shapes = [(2, 3, 4)]
        output_shape = (2, 3, 4)

        def reference_fn(args):
            return np.power(args[0], 0.7)

        def operation_fn(args):
            power = ops.Power(exponent=0.7)
            return power(args[0])

        self._test_op(input_shapes, output_shape, reference_fn,
                      operation_fn, positive=True)

    def test_maximum(self):
        input_shapes = [(2, 3), (2, 3), (2, 3)]
        output_shape = (2, 3)

        def reference_fn(args):
            return functools.reduce(lambda x, y: np.maximum(x, y), args)

        def operation_fn(args):
            maximum = ops.Maximum()
            return maximum(args)

        self._test_op(input_shapes, output_shape, reference_fn,
                      operation_fn, positive=False)


    def test_relu(self):
        input_shapes = [(2, 5)]
        output_shape = (2, 5)

        def reference_fn(args):
            return np.maximum(args[0], 0.0)

        def operation_fn(args):
            maximum = ops.ReLU()
            return maximum(args[0])

        self._test_op(input_shapes, output_shape, reference_fn,
                      operation_fn, positive=False)

    def test_reduce_max(self):
        input_shapes = [(2, 3, 4)]
        output_shape = (1)

        def reference_fn(args):
            return np.max(args[0])

        def operation_fn(args):
            reduce_max = ops.ReduceMax()
            return reduce_max(args[0])

        self._test_op(input_shapes, output_shape, reference_fn,
                      operation_fn, positive=False)

    def test_hinge_loss(self):
        input_shapes = [(10)]
        output_shape = (1)

        labels = [0, 3, 6]

        def reference_fn_label(label, args):
            return np.sum(np.maximum(args[0] - args[0][label] + 1.0, 0.0))/len(args[0])

        reference_fns = [lambda args: reference_fn_label(label, args) for label in labels]

        def operation_fn_label(label, args):
            hinge = ops.HingeLoss(label)
            return hinge(args[0])
        
        operation_fns = [lambda args: operation_fn_label(label, args) for label in labels]

        for operation_fn, reference_fn in zip(operation_fns, reference_fns):
            self._test_op(input_shapes, output_shape, reference_fn,
                        operation_fn, positive=False)

    def test_hinge_uses_upstream(self):
        input_shapes = [(10)]
        output_shape = (1)

        labels = [0, 3, 6]

        scaleFactor = 10.0

        def reference_fn_label(label, args):
            return np.sum(np.maximum(args[0] - args[0][label] + 1.0, 0.0))/len(args[0])

        reference_fns = [lambda args: scaleFactor * reference_fn_label(label, args) for label in labels]

        def operation_fn_label(label, args):
            hinge = ops.HingeLoss(label)
            mul = ops.ScalarMultiply()
            scaleFactorVariable = Variable(scaleFactor)

            return mul(scaleFactorVariable, hinge(args[0])) 
        
        operation_fns = [lambda args: operation_fn_label(label, args) for label in labels]

        for operation_fn, reference_fn in zip(operation_fns, reference_fns):
            self._test_op(input_shapes, output_shape, reference_fn,
                        operation_fn, positive=False)
    def test_chained_ops(self):
        input_shapes = [(2, 3), (3, 4), (2, 4)]
        output_shape = (1)

        def reference_fn(args):
            return np.max(args[2]+np.exp(np.dot(args[0], args[1])))

        def operation_fn(args):
            matmul = ops.MatrixMultiply()
            exp = ops.Exp()
            add = ops.VariableAdd()
            reduce_max = ops.ReduceMax()

            return reduce_max(add([args[2], exp(matmul(args[0], args[1]))]))

        self._test_op(input_shapes, output_shape, reference_fn,
                      operation_fn, positive=False)

    def test_reuse_vars(self):
        input_shapes = [(2, 3)]
        output_shape = (2, 3)

        def reference_fn(args):
            return args[0]*args[0]

        def operation_fn(args):
            mul = ops.VariableMultiply()
            return mul([args[0], args[0]])

        self._test_op(input_shapes, output_shape, reference_fn,
                      operation_fn, positive=False)

    def test_graph(self):
        input_shapes = [(1), (1), (1)]
        output_shape = (1)

        def reference_fn(args):
            x = args[0] * args[1]
            y = x + args[2]
            z = y*args[1]
            w = z + x
            return w

        def operation_fn(args):
            mul1 = ops.VariableMultiply()
            add1 = ops.VariableAdd()
            mul2 = ops.VariableMultiply()
            add3 = ops.VariableAdd()

            x = mul1([args[0], args[1]])
            y = add1([x, args[2]])
            z = mul2([y, args[1]])
            w = add3([z, x])

            return w

        self._test_op(input_shapes, output_shape, reference_fn,
                      operation_fn, positive=False)

    def test_large_graph(self):
        input_shapes = [(2, 3), (3, 4), (2, 4), (2, 4), (4, 3)]
        output_shape = (2, 3)

        def reference_fn(args):
            x = np.maximum(args[2], np.dot(args[0], args[1]))
            y = x*x
            z = np.power(y, 1.3)
            w = z + args[3]
            a = np.dot(w, args[4])
            b = args[0] + a
            return b

        def operation_fn(args):
            matmul1 = ops.MatrixMultiply()
            maximum = ops.Maximum()
            mul = ops.VariableMultiply()
            power = ops.Power(1.3)
            add1 = ops.VariableAdd()
            matmul2 = ops.MatrixMultiply()
            add2 = ops.VariableAdd()

            x = maximum([args[2], matmul1(args[0], args[1])])
            y = mul([x, x])
            z = power(y)
            w = add1([z, args[3]])
            a = matmul2(w, args[4])
            b = add2([args[0], a])

            return b

        self._test_op(input_shapes, output_shape, reference_fn,
                      operation_fn, positive=False)

    def _test_op(self, input_shapes, output_shape, reference_fn, operation_fn, positive=False):
        forward_diff = test_forward_random(
            input_shapes, reference_fn, operation_fn, positive=positive)
        self.assertLessEqual(forward_diff, 1e-5)

        backward_diff = test_backward_random(
            input_shapes, output_shape, reference_fn, operation_fn, positive=positive)
        self.assertLessEqual(backward_diff, 1e-5)


if __name__ == '__main__':
    unittest.main()
