import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import h5py
import plotly.graph_objects as go
from matplotlib.animation import FuncAnimation

X, y = make_blobs(n_samples=100, n_features=2, centers=2, random_state=0)

y = y.reshape(y.shape[0], 1)

print("X dimensions : ", X.shape)
print("y dimensions : ", y.shape)

#dataset visualisation
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='summer')
plt.show()