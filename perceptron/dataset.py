import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import numpy as np

def generate_dataset(n_samples=100, random_state=0):
    X, y = make_blobs(n_samples=n_samples,
                    n_features=2,
                    centers=2,
                    random_state=random_state)
    y = y.reshape((y.shape[0], 1))
    return X, y


def plot_dataset(X, y):
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='summer')
    plt.xlabel('x1 (Width)')
    plt.ylabel('x2 (Length)')
    plt.title('Plant Dataset (Synthetic)')
    plt.show()

def plot_decision_boundary(X, y, W, b):
    """
    Plots dataset scatter with learned decision boundary line.

    Inputs:
    X - Numpy matrix of input values (n_samples, n_features)
    y - Numpy array of true labels (n_samples, 1)
    W - Learned weights (n_features, 1)
    b - Learned bias (scalar)
    """
    #creating boundary points
    X0_boundary = np.linspace(X[: ,0].min() -1, X[:,0].max() + 1, 100)
    #computes points where Z = 0 & w1*x1 + w2*x2 + b = 0
    #i.e. w2 * X2 = -w1 * X1 - b --> X2 = (-w1 * X1 - b) / w2
    X1_boundary = (- W[0] * X0_boundary - b) / W[1]
    #colouring datapoints
    colors_scatter = ['orange' if label == 0 else 'blue' for label in y.ravel()] #y.ravel() flattens to 1D vector
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:,0], X[:,1], c=colors_scatter, label='Original Data (0:orange, 1 : blue)')
    plt.plot(X0_boundary, X1_boundary, c='purple', lw=3, label='Decision Boundary')
    plt.xlabel('X1 - feature 1 Width')
    plt.ylabel('X2 - feature 2 Length')
    plt.title('Features Decision Boundary')
    plt.legend ()
    plt.xlim(X[: , 0].min() - 1 , X[: , 0].max() + 1)
    plt.ylim(X[: , 1].min() - 1 , X[: , 1].max() + 1)
    plt.show()

if __name__ == '__main__':
    X, y = generate_dataset()
    print('X dimensions:', X.shape)   # (100, 2)
    print('y dimensions:', y.shape)   # (100, 1)
    plot_dataset(X, y)
