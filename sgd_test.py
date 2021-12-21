from tensor import Tensor
import ops_impl as ops
import numpy as np
import functools
from sgd import SGD
from sklearn.datasets import fetch_openml

# load mnist data:
def load_mnist():
    mnist = fetch_openml('mnist_784', data_home='./data', as_frame=False)
    mnist.target = np.array([int(t) for t in mnist.target])
    return mnist
    mnist.target = np.array([int(t) for t in mnist.target])
    mnist.target_one_hot = np.zeros((len(mnist.target), 10))
    row_indices = np.arange(len(mnist.target))
    mnist.target_one_hot[row_indices, mnist.target] = 1.0


def loss_fn(params, data):
    '''computes hinge for linear classification of MNIST digits.
    
    args:
        params: list containing [weights, bias]
            where weights is a 10x784 tensor and
            bias is a scalar tensor.
        data: list containing [features, label]
            where features is a 784 dimensional numpy array
            and label is an integer
        
    returns:
        loss, correct
            where loss is a tensor representing the hinge loss
            of the 10-dimenaional scores where
            scores[i] = dot(weights[i] , features) + bias
            and correct is 1.0 if scores[label] is the largest score
            and zero otherwise.
    '''

    # weights, bias = params
    features, label = data
    features = Tensor(features)

    scores = get_scores(features, params)#ops.matmul(weights, features) + bias

    loss = ops.HingeLoss(label)(scores)

    correct = np.argmax(scores.data) == label

    return loss, correct

def get_scores(features, params):
    
    weights, bias = params
    return ops.matmul(weights, features) + bias

    weights1, bias1, weights2, bias2, weights3, bias3 = params

    act1 = ops.relu(ops.matmul(weights1, features) + bias1)

    act2 = ops.relu(ops.matmul(weights2, act1) + bias2)

    scores = ops.matmul(weights3, act2) + bias3

    return scores

def get_normal(shape):
    return np.random.normal(np.zeros(shape))

def test_sgd(learning_rate, epochs, mnist):
    running_accuracy = 0.0
    it = 0
    params = [Tensor(np.zeros((10, 784))), Tensor(np.zeros((10, 1)))]
    # params1 = [Tensor(get_normal((100, 784))), Tensor(get_normal((100, 1)))]
    # params2 = [Tensor(get_normal((50, 100))), Tensor(get_normal((50, 1)))]
    # params3 = [Tensor(get_normal((10, 50))), Tensor(get_normal((10, 1)))]
    # params = params1+params2+params3
    for it in range(epochs*60000):
        data = [mnist.data[it % 60000].reshape(-1, 1)/255.0, mnist.target[it % 60000]]
        params, correct = SGD(loss_fn, params, data, learning_rate)
        running_accuracy += (correct - running_accuracy)/(it + 1.0)
        if (it+1) % 10000 == 0:
            print("it: {}, running train accuracy: {}".format(it+1, running_accuracy))

    running_accuracy = 0.0
    for it in range(10000):
        data = [mnist.data[it + 60000].reshape(-1, 1)/255.0, mnist.target[it + 60000]]
        loss, correct = loss_fn(params, data)
        running_accuracy += (correct - running_accuracy)/(it + 1.0)
    print("eval accuracy: ", running_accuracy)


if __name__ == '__main__':
    mnist = load_mnist()
    test_sgd(0.01, 2, mnist)