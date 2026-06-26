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
    Computes the forward propagation.

    Arguments:
        X -- feature matrix (n_samples, n_features)
        W -- weight vector (n_features, 1)
        b -- bias (scalar)

    Returns:
        A -- activation vector (n_samples, 1)
    """
    Z = np.dot(X, W) + b
    A = 1 / (1 + np.exp(-Z))
    return A

def log_loss(A, y):
    """
    Computes the binary cross-entropy cost.

    Arguments:
        A -- activation vector of predictions (n_samples, 1)
        y -- true labels vector (n_samples, 1)

    Returns:
        loss -- scalar log-loss value
    """
    m = len(y)
    epsilon = 1e-15
    loss = (1 / m) * np.sum(-y * np.log(A + epsilon) - (1 - y) * np.log(1 - A + epsilon))
    return loss

def gradients(A, X, y):
    """
    Computes gradients of cost with respect to W and b.

    Arguments:
        A -- activation vector of predictions (n_samples, 1)
        X -- feature matrix (n_samples, n_features)
        y -- true labels vector (n_samples, 1)

    Returns:
        dW -- gradient of cost with respect to W (n_features, 1)
        db -- gradient of cost with respect to b (scalar)
    """
    m = len(y)
    dW = (1 / m) * np.dot(X.T, (A - y))
    db = (1 / m) * np.sum(A - y)
    return dW, db

def update(dW, db, W, b, learning_rate):
    """
    Updates parameters W and b using gradient descent.

    Arguments:
        dW -- gradient of W (n_features, 1)
        db -- gradient of b (scalar)
        W -- weight vector (n_features, 1)
        b -- bias (scalar)
        learning_rate -- step size for gradient descent

    Returns:
        W -- updated weight vector (n_features, 1)
        b -- updated bias (scalar)
    """
    W = W - learning_rate * dW
    b = b - learning_rate * db
    return W, b

def predict(X, W, b):
    """
    Predicts classes (0 or 1) using learned parameters.

    Arguments:
        X -- feature matrix (n_samples, n_features)
        W -- weight vector (n_features, 1)
        b -- bias (scalar)

    Returns:
        predictions -- boolean/binary array (n_samples, 1)
    """
    A = model(X, W, b)
    return A >= 0.5

def artificial_neuron(X, y, learning_rate=0.1, n_iter=100):
    """
    Trains the artificial neuron on the provided dataset.

    Arguments:
        X -- feature matrix (n_samples, n_features)
        y -- true labels vector (n_samples, 1)
        learning_rate -- learning rate alpha (scalar)
        n_iter -- number of iterations (integer)

    Returns:
        W -- trained weights (n_features, 1)
        b -- trained bias (scalar)
        loss_history -- list of loss values at each iteration
    """
    W, b = init_neuron(X)
    loss_history = []
    
    for i in range(n_iter):
        A = model(X, W, b)
        loss = log_loss(A, y)
        loss_history.append(loss)
        dW, db = gradients(A, X, y)
        W, b = update(dW, db, W, b, learning_rate)
        
    return W, b, loss_history