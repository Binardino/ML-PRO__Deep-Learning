from utilities import load_data
from perceptron.multi_layer_perceptron import neural_network, predict, normalise_data , flatten_data
from sklearn.metrics import accuracy_score

def main():
    X_train, y_train, X_test, y_test = load_data()

    X_train_reshape, X_test_reshape = flatten_data(X_train, X_test)
        
    X_train_norm, X_test_norm = normalise_data(X_train_reshape, X_test_reshape)

    parameters, train_loss, train_acc = neural_network(X_train_norm.T, y_train.T, hidden_dims=[32], learning_rate=0.001, n_iter=10000)

    print("\nFinal parameters:")
    print("W1 shape:", parameters['W1'].shape)
    print("W2 shape:", parameters['W2'].shape)
    print("Final train loss:", train_loss[-1])
    print("Final train accuracy:", train_acc[-1])

if __name__ == "__main__":
    main()