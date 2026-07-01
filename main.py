from perceptron.dataset import generate_dataset, plot_decision_boundary
from perceptron.neuron import artificial_neuron, predict
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def main():
    X, y = generate_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    W, b, loss_history = artificial_neuron(X_train, y_train, learning_rate=0.1, n_iter=10000)
    print("\nFinal parameters:")
    print("W:", W)
    print("b:", b)

    #accuracy
    y_pred   = predict(X_test, W, b)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"accuracy score : {accuracy * 100:.2f}%")

    # Frontière de décision
    plot_decision_boundary(X, y, W, b)

if __name__ == "__main__":
    main()
