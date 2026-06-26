import numpy as np

def init_neuron(X):
    """
    Initialise W and b parameters for the neuron.

    Arguments:
        X -- feature matrix (n_samples, n_features)

    Returns:
        W -- weight vector (n_features, 1), randomly initialized
        b -- bias (scalar), randomly initialized
    """
    n_features = X.shape[1]
    W          = np.random.randn(n_features, 1)  # one weight per feature
    b          = np.random.randn(1)               # single bias

    return W, b

def model(X, W, b):
    """
    Compute neuron outputs - forward propagation
    Input : 
    X - Numpy matrix of input values (n_samples, n_features)
    W - Numpy array of Weights for each input (n_features, 1)
    b - Numpy array of bias (scalar)

    Return :
    A - Numpy array of Activations (n_samples, 1)

    """
    Z = X.dot(W) + b            # Linear combination - dot product
    A = 1 / ( 1 + np.exp(-Z))   # Sigmoid function

    return A

def log_loss(A, y):
    """
    Compute the binary cross-entropy cost - i.e. loss function (Log Loss)

    Inputs :
    A - Numpy array - activation vector of Predictions (n_samples, 1)
    y - Numpy array of real y values (n_samples, 1)

    Output:
    Loss results - scalar value of cost function
    """
    m       = len(y)
    epsilon = 1e-15  # prevents log(0) = -inf when A is exactly 0 or 1

    loss    = (1/m) * np.sum(-y * np.log(A + epsilon)           # cost for positive class (y=1)
                          - (1 - y) * np.log(1 - A + epsilon))  # cost for negative class (y=0)

    return loss


def gradient(A, X, y):
    m  = len(y)
    dW = (1/m) * np.dot(X.T, (A - y))
    db = (1/M) * np.sum((A - y))

    return dw, db