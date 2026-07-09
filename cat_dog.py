from utilities import load_data
from perceptron.neuron import artificial_neuron, predict, normalise_data , flatten_data
from sklearn.metrics import accuracy_score

def main():
    X_train, y_train, X_test, y_test = load_data()

    X_train_reshape, X_test_reshape = flatten_data(X_train, X_test)
        
    X_train_norm, X_test_norm = normalise_data(X_train_reshape, X_test_reshape)

    W, b, loss_history = artificial_neuron(X_train_norm, y_train, learning_rate=0.001, n_iter=10000)
    print("\nFinal parameters:")
    print("W shape:", W.shape)
    print("b:", b)
    print("Final loss:", loss_history[-1])

    #accuracy
    y_pred   = predict(X_test_norm, W, b)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"accuracy score : {accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()