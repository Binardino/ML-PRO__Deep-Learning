import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score

def init_neuron(n0, n1, n2):
    """
    Initialise W and b parameters for a 2-layer network (1 hidden + 1 output layer).

    Arguments:
        n0 -- number of input features
        n1 -- number of neurons in the hidden layer
        n2 -- number of neurons in the output layer

    Returns:
        parameters -- dict containing W1, b1, W2, b2, randomly initialized
    """
    W1 = np.random.randn(n1, n0)
    b1 = np.random.randn(n1, 1)
    W2 = np.random.randn(n2, n1)
    b2 = np.random.randn(n2, 1)

    parameters = {
        'W1' : W1,
        'b1' : b1,
        'W2' : W2,
        'b2' : b2
    }

    return parameters

def forward_propagation(X, parameters):
    """
    Compute the activations of both layers - forward propagation.

    Arguments:
        X          -- input feature matrix (n0, m)
        parameters -- dict containing W1, b1, W2, b2

    Returns:
        activations -- dict containing A1 (hidden layer output) and A2 (final output)
    """
    W1, W2 = parameters['W1'], parameters['W2']
    b1, b2 = parameters['b1'], parameters['b2']

    Z1 = W1.dot(X) + b1
    A1 = 1 / (1 + np.exp(-Z1))

    # A1 (not Z1) feeds the next layer: the sigmoid is what keeps the
    # 2-layer stack from collapsing into a single linear transformation.
    Z2 = W2.dot(A1) + b2
    A2 = 1 / (1 + np.exp(-Z2))

    activations = {
        'A1' : A1,
        'A2' : A2
    }

    return activations

def back_propagation(X, y, parameters, activations):
    """
    Compute gradients of the cost function w.r.t. W1, b1, W2, b2 - backpropagation.

    Arguments:
        X           -- input feature matrix (n0, m)
        y           -- true labels (n2, m)
        parameters  -- dict containing W1, b1, W2, b2
        activations -- dict containing A1, A2 (from forward_propagation)

    Returns:
        gradients -- dict containing dW1, db1, dW2, db2
    """
    A1 = activations['A1']
    A2 = activations['A2']
    W2 = parameters['W2']
    m = y.shape[1]

    # Output layer error (identical form to the single-neuron gradient: A - y)
    dZ2 = A2 - y
    dW2 = 1 / m * dZ2.dot(A1.T)
    # keepdims=True preserves shape (n2, 1) instead of collapsing to (n2,),
    # which would break broadcasting when updating b2
    db2 = 1 / m * np.sum(dZ2, axis = 1, keepdims=True)

    # Propagate the error back through W2, then apply the local sigmoid
    # derivative A1 * (1 - A1) to get the hidden layer's error term
    dZ1 = np.dot(W2.T, dZ2) * A1 * (1 - A1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)

    gradients = {
        'dW1' : dW1,
        'db1' : db1,
        'dW2' : dW2,
        'db2' : db2
    }

    return gradients

def update(parameters, gradients, learning_rate):
    """
    Update W1, b1, W2, b2 using gradient descent step.

    Arguments:
        parameters    -- dict containing current W1, b1, W2, b2
        gradients     -- dict containing dW1, db1, dW2, db2
        learning_rate -- scalar step size

    Returns:
        parameters -- updated dict of W1, b1, W2, b2
    """
    W1 = parameters['W1'] - learning_rate * gradients['dW1']
    b1 = parameters['b1'] - learning_rate * gradients['db1']
    W2 = parameters['W2'] - learning_rate * gradients['dW2']
    b2 = parameters['b2'] - learning_rate * gradients['db2']

    parameters = {
        'W1' : W1,
        'b1' : b1,
        'W2' : W2,
        'b2' : b2
    }

    return parameters
        