
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
    plt.xlabel('x1 (Largeur)')
    plt.ylabel('x2 (Longueur)')
    plt.title('Dataset des Plantes (Synthétique)')
    plt.show()


if __name__ == '__main__':
    X, y = generate_dataset()
    print('dimensions de X:', X.shape)  # (100, 2)
    print('dimensions de y:', y.shape)  # (100, 1)
    plot_dataset(X, y)
