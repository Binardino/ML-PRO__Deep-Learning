from utilities import load_data
from perceptron.dataset import generate_dataset, plot_decision_boundary
from perceptron.neuron import artificial_neuron, predict, flatten_data
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

#update main for animal image dataset
def main():
# used for percepton model
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_test, y_train, y_test = load_data()
    X_train_reshape, X_test_reshape = flatten_data(X_train, X_test)
    W, b, loss_history = artificial_neuron(X_train_reshape, y_train, learning_rate=0.1, n_iter=10000)
    print("\nFinal parameters:")
    print("W:", W)
    print("b:", b)
    print("Final loss:", loss_history[-1])

    #accuracy
    y_pred   = predict(X_test_reshape, W, b)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"accuracy score : {accuracy * 100:.2f}%")

    # Decision boundary - used for percepton model
    #plot_decision_boundary(X_train, y_train, W, b)

if __name__ == "__main__":
    main()
