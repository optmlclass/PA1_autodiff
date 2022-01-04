# Programming  Assignment 1: Automatic Differentation

Note: this assignment is complicated! Do not start it at the last minute! You may
want to read over the code to get an understanding of what you should be doing
before diving in!

Your code will be evaluated in two ways:
1. You must pass the automated tests that are invoked by running:
```
python3 numerical_test.py
```
This file will compare your automatic differentiation results to the slower
numerical differentiation baseline. You are encouraged to read the tests, and 
potentially make copies for debugging your code. However, you MUST NOT modify
this file.

Our solution implementation produces the following output on a 2016 macbook pro:
```
$ python3 numerical_test.py 
.....................
----------------------------------------------------------------------
Ran 21 tests in 0.048s

OK
```

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
Although you may make arbitrary changes when debugging, the file you turn in 
is NOT allowed to edit the "load_mnist" or "train_mnist" functions in 
`sgd_test.py` except for the point marked. Your code must actually implement SGD.
Do not attempt to hard-code some "good" weights into your implementation, or change
the loss function to return small values.

## setting up your environment (assumes python3 is already installed)

If you want to use a virtual environment (recommended, but not required):
```
python3 -m venv autograd
source autograd/bin/activate
```

Regardless, make sure you have the following dependencies:
```
pip3 install numpy
pip3 install sklearn
```


You need to write code to finish functions in the files listed below.
Check the docstring below each function definition for a description of what
the function should accomplish.
Note that some of the "backward_call" and "forward_call" methods do not  have
docstrings. For these, check docstrings for their class definition to see
what the relevant operation is computing, and check the docstrings in the 
Variable and Operation classes to see overall how these functions should work.

This starter code assumes that the final output of any computation graph is a 
scalar so that the total derivative is actually a gradient. As a result, the
word "gradient" or "grad" is used in variable names rather than "derivative".



operation.py:
    finish the "backward" function.

variable.py:
    finish the "backward" function.

ops_imply.py:
    finish the "forward_call" and "backward_call" functions in the following classes:
        VariableAdd
        VariableMultiply
        ScalarMultiply
        MatrixMultiply
        HingeLoss

    The other Operation implementations in this file are there to provide examples.

sgd.py:
    write the function "SGD".

sgd_test.py:
    write the function "loss_fn".





