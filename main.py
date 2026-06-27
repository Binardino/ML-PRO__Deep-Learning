from perceptron.dataset import generate_dataset
from perceptron.neuron import artificial_neuron

def main():
    X, y = generate_dataset()
    W, b, loss_history = artificial_neuron(X, y, learning_rate=0.1, n_iter=10000)
    print("\nFinal parameters:")
    print("W:", W)
    print("b:", b)

if __name__ == "__main__":
    main()
