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

def log_loss(A, y):
    """
    Compute the binary cross-entropy cost - log loss function
    Inputs :
    A - Np array - activation vectors
    y - Np array - real y vales

    Outputs:
    Loss results - scalar value of cost function
    """
    m       = y.shape[1]
    epsilon = 1e-15

    loss = (1 / m) * np.sum(-y * np.log(A + epsilon)
                            - (1 - y) * np.log(1 - A + epsilon)
    ) 

    return loss


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

def predict(X, parameters):
    """
    Predict class 0/1 for input data X using the trained 2-layer network.

    Arguments:
        X          -- input feature matrix (n0, m)
        parameters -- dict containing trained W1, b1, W2, b2

    Returns:
        Boolean array (n2, m): True = class 1, False = class 0 (threshold 0.5 on A2)
    """
    activations = forward_propagation(X, parameters)
    A2 = activations['A2']
    return A2 >= 0.5

def flatten_data(X_train, X_test):
    """
    Flatten multi-dimensional input data into 2D matrices (m, n_features).

    Arguments:
        X_train -- training data, any shape (m_train, d1, d2, ...)
        X_test  -- test data, any shape (m_test, d1, d2, ...)

    Returns:
        X_train_reshape -- flattened training data (m_train, n_features)
        X_test_reshape  -- flattened test data (m_test, n_features)

    Note: output is row-per-example (m, n_features). Transpose to
    (n_features, m) before passing into neural_network().
    """
    m_train, m_test = X_train.shape[0], X_test.shape[0]
    X_train_reshape, X_test_reshape = X_train.reshape(m_train, -1), X_test.reshape(m_test, -1)

    print('X_train_reshape shapes =', X_train_reshape.shape)
    print('X_test_reshape shapes  =', X_test_reshape.shape)

    return X_train_reshape, X_test_reshape

def normalise_data(X_train, X_test):
    """
    Apply Min-Max normalization using the training set's min/max only,
    to avoid data leakage from the test set.

    Arguments:
        X_train -- flattened training data (m_train, n_features)
        X_test  -- flattened test data (m_test, n_features)

    Returns:
        X_train_norm -- normalized training data, same shape as X_train
        X_test_norm  -- normalized test data, same shape as X_test
    """
    X_train_norm = (X_train - X_train.min()) / (X_train.max() - X_train.min())
    X_test_norm  = (X_test  - X_train.min()) / (X_train.max() - X_train.min())

    print(f"X_train bounds : min={X_train_norm.min():.2f}, max={X_train_norm.max():.2f}")
    print(f"X_test bounds  : min= {X_test_norm.min():.2f}, max={X_test_norm.max():.2f}")

    return X_train_norm, X_test_norm

def neural_network(X_train, y_train, n1, learning_rate=0.01, n_iter=1000):
    """
    Train a 2-layer neural network (1 hidden + 1 output layer) via gradient descent.

    Arguments:
        X_train       -- training feature matrix (n0, m)
        y_train       -- training labels (n2, m)
        n1            -- number of neurons in the hidden layer
        learning_rate -- step size for gradient descent (default 0.01)
        n_iter        -- number of training iterations (default 1000)

    Returns:
        parameters -- trained dict of W1, b1, W2, b2
        train_loss -- training loss sampled every 10 iterations
        train_acc  -- training accuracy sampled every 10 iterations

    Plots: loss curve and accuracy curve (train) at end of training.
    """
    #init
    n0 = X_train.shape[0]
    n2 = y_train.shape[0]
    parameters = init_neuron(n0, n1, n2)

    train_loss = []
    train_acc = []

    for i in range(n_iter):
        activations = forward_propagation(X_train, parameters)
        gradients   = back_propagation(X_train, y_train, parameters, activations)
        parameters  = update(parameters, gradients, learning_rate)

        if i %10 == 0:
            train_loss.append(log_loss(activations['A2'], y_train))
            y_pred = predict(X_train, parameters)
            current_accuracy = accuracy_score(y_train.flatten(), y_pred.flatten())
            train_acc.append(current_accuracy)

    plt.figure(figsize=(14,4))

    plt.subplot(1,2,1)
    plt.plot(train_loss, label='train_loss')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(train_acc, label='train_acc')
    plt.legend()

    plt.show()

    return parameters, train_loss, train_acc