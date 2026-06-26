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