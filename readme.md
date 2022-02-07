# Programming  Assignment 1: Automatic Differentation

Note: this assignment is complicated! Do not start it at the last minute! You may
want to read over the code to get an understanding of what you should be doing
before diving in!

## Instructions

You need to write code to finish functions in the files listed below.
Check the docstring below each function definition for a description of what
the function should accomplish  (a "docstring" in python is a long comment
wrapped in triple-quotes (''') at the top of a file, class or function
declaration that describes what the  file, class or function does. They
are used by some automatic documentation tools).
Note that some of the `backward_call` and `forward_call` methods do not  have
docstrings. For these, check docstrings for their class definition to see
what the relevant operation is computing, and check the docstrings in the 
`Variable` and `Operation` classes to see overall how these functions should work.

This starter code assumes that the final output of any computation graph is a 
scalar so that the total derivative is actually a gradient. As a result, the
word "gradient" or "grad" is used in variable names rather than "derivative".



operation.py:
finish the `backward` function.

variable.py:
finish the `backward` function.

ops_imply.py:
finish the `forward_call` and `backward_call` functions in the following classes:
- `VariableAdd`
- `VariableMultiply`
- `ScalarMultiply`
- `MatrixMultiply`
- `HingeLoss`

The other `Operation` implementations in this file are there to provide examples.

sgd.py:
write the function `SGD`.

sgd_test.py:
write the function `loss_fn`.


## Setting up your environment (assumes python3 is already installed)

This assignment needs [numpy](https://numpy.org/) and [sklearn](https://scikit-learn.org/stable/), which are both standard python packages
that can be installed via whatever package manager you use.

If you want to use a [virtual environment](https://docs.python.org/3/tutorial/venv.html) with pip:
```
python3 -m venv autograd
source autograd/bin/activate
pip3 install numpy
pip3 install sklearn
```

If you use [anaconda python](https://www.anaconda.com/products/individual), they are probably already installed.


## Evaluation

Your code will be evaluated in two ways:
1. You must pass the automated tests that are invoked by running:
```
python3 numerical_test.py
```
This file will compare your automatic differentiation results to the slower
numerical differentiation baseline. You are encouraged to read the tests, and 
potentially make copies and changes for debugging your code. However, the 
submission MUST NOT modify this file.

Our solution implementation produces the following output on a 2016 macbook pro:
```
$ python3 numerical_test.py 
.....................
----------------------------------------------------------------------
Ran 21 tests in 0.048s

OK
```

Each test is worth 2 points, for a total of 44 points from the automated tests.

For reference, before writing any code, the automated tests should output:
```
$ python3 numerical_test.py 
EEEEEEEEEEEEEEEEEEEEE
======================================================================
ERROR: test_add (__main__.TestAutograd)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/Users/ashok/Projects/ec500hw/spring2022/numerical_test.py", line 108, in test_add
    self._test_op(input_shapes, output_shape, reference_fn,
  File "/Users/ashok/Projects/ec500hw/spring2022/numerical_test.py", line 468, in _test_op
    forward_diff = test_forward_random(
  File "/Users/ashok/Projects/ec500hw/spring2022/numerical_test.py", line 88, in test_forward_random
    analytic = operation_fn(tensors).data
  File "/Users/ashok/Projects/ec500hw/spring2022/numerical_test.py", line 107, in operation_fn
    return add(args)
  File "/Users/ashok/Projects/ec500hw/spring2022/operation.py", line 27, in __call__
    return self.forward(*args, **kwargs)
  File "/Users/ashok/Projects/ec500hw/spring2022/operation.py", line 32, in forward
    output = self.forward_call(*args, **kwargs)
  File "/Users/ashok/Projects/ec500hw/spring2022/ops_impl.py", line 70, in forward_call
    raise NotImplementedError
NotImplementedError

~~~~~~~~~~ many more error messages printed here ~~~~~~~~~~~~


======================================================================
ERROR: test_tensordot (__main__.TestAutograd)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/Users/ashok/Projects/ec500hw/spring2022/numerical_test.py", line 254, in test_tensordot
    self._test_op(input_shapes, output_shape, reference_fn,
  File "/Users/ashok/Projects/ec500hw/spring2022/numerical_test.py", line 472, in _test_op
    backward_diff = test_backward_random(
  File "/Users/ashok/Projects/ec500hw/spring2022/numerical_test.py", line 75, in test_backward_random
    output.backward(downstream_grad)
  File "/Users/ashok/Projects/ec500hw/spring2022/variable.py", line 58, in backward
    raise NotImplementedError
NotImplementedError

----------------------------------------------------------------------
Ran 21 tests in 0.017s

FAILED (errors=21)
```

It is strongly recommended that you pass these tests BEFORE attempting to implement SGD.

2. You must obtain >89% eval accuracy on MNIST in a reasonable amount of time (<5 minutes) as measured
by `sgd_test.py'. Our solution implementation runs in about a minute:
```
$ time python sgd_test.py
loading mnist data....
done!
training linear classifier...
iteration: 10000, current train accuracy: 0.8738000000000002
iteration: 20000, current train accuracy: 0.883700000000004
iteration: 30000, current train accuracy: 0.8898333333333336
iteration: 40000, current train accuracy: 0.8919750000000018
iteration: 50000, current train accuracy: 0.8938599999999982
iteration: 60000, current train accuracy: 0.8975666666666641
iteration: 70000, current train accuracy: 0.9004142857142788
iteration: 80000, current train accuracy: 0.9015749999999935
iteration: 90000, current train accuracy: 0.903099999999996
iteration: 100000, current train accuracy: 0.9034999999999993
iteration: 110000, current train accuracy: 0.9039181818181822
iteration: 120000, current train accuracy: 0.9053583333333246
running evaluation...
eval accuracy:  0.9132000000000016

real    1m3.388s
user    0m55.799s
sys     0m1.539s
```
Although you may certainly make arbitrary changes when debugging, your submission should not
edit sgd_test.py except for the point marked "YOUR CODE HERE".

A correct SGD implementation is worth 6 points, for a total of 50 points possible on this assignment.





