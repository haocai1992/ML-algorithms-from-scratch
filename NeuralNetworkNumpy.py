# Numpy implementation of Neural Network
# References:
# https://www.kaggle.com/code/wwsalmon/simple-mnist-nn-from-scratch-numpy-no-tf-keras/notebook
# https://www.youtube.com/watch?v=w8yWXqWQYmU
# https://victorzhou.com/blog/intro-to-neural-networks/

#   Neural network with:
#   - an input layer with 784 features
#   - a hidden layer with 10 neurons with Relu activation
#   - an output layer with 10 neuron with softmax activation
#   - loss function: cross-entropy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class NeuralNetworkNumpy:
    def __init__(self):
        """
        X: [[x11, x12, ..., x1n], ...[xm1, xm2, ..., xmn]]
        y: [y1, y2, ..., ym]; y = 0 to 9 (10 classes)
        """
        self.X = None # 784 x m
        self.y = None
        # hidden layer
        self.W1 = None # 10 x 784
        self.b1 = None # 10 x 1
        # output layer
        self.W2 = None # 10 x 10
        self.b2 = None # 10 x 1

    def train(self, X, y, n_iterations=1000, learning_rate=0.05):
        W1, b1, W2, b2 = self.init_params()
        for i in range(n_iterations):
            Z1, A1, Z2, A2 = self.forward_prop(X, W1, b1, W2, b2)
            dW1, db1, dW2, db2 = self.back_prop(X, y, W1, W2, Z1, A1, Z2, A2) # Here, dW1 means dL_dW1
            W1, b1, W2, b2 = self.update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate)
        self.W1, self.b1, self.W2, self.b2 = W1, b1, W2, b2
        return W1, b1, W2, b2

    def predict(self, X):
        y = self.forward_prop(X, self.W1, self.b1, self.W2, self.b2)[-1]
        print(y)
        return np.argmax(y, 0)[0]

    def init_params(self):
        W1 = np.random.rand(10, 784) - 0.5
        b1 = np.random.rand(10, 1) - 0.5
        W2 = np.random.rand(10, 10) - 0.5
        b2 = np.random.rand(10, 1) - 0.5
        return W1, b1, W2, b2
    
    def forward_prop(self, X, W1, b1, W2, b2):
        A0 = X # 784 x m
        Z1 = np.dot(W1, A0) + b1
        A1 = self.relu(Z1)
        Z2 = np.dot(W2, A1) + b2
        A2 = self.softmax(Z2)
        return Z1, A1, Z2, A2

    def back_prop(self, X, y, W1, W2, Z1, A1, Z2, A2):
        m, n = X.shape
        one_hot_y = self.one_hot(y)
        dZ2 = A2 - one_hot_y # this is the derivative of softmax + cross-entropy loss. deduction see: https://www.adeveloperdiary.com/data-science/deep-learning/neural-network-with-softmax-in-python/
        dW2 = 1/m * np.dot(dZ2, A1.T)
        db2 = 1/m * np.sum(dZ2)
        dZ1 = np.dot(W2.T, dZ2) * self.relu_deriv(Z1)
        dW1 = 1/m * np.dot(dZ1, X.T)
        db1 = 1/m * np.sum(dZ1)
        return dW1, db1, dW2, db2
    
    def update_params(self, W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate):
        W1 -= dW1 * learning_rate
        b1 -= db1 * learning_rate
        W2 -= dW2 * learning_rate
        b2 -= db2 * learning_rate
        return W1, b1, W2, b2

    def relu(self, Z):
        return np.maximum(Z, 0)
    
    def relu_deriv(self, Z):
        return Z>0
    
    def softmax(self, Z):
        A = np.exp(Z) / sum(np.exp(Z))
        return A
    
    def one_hot(self, y):
        one_hot_y = np.zeros((y.size, y.max() + 1))
        one_hot_y[np.arange(y.size), y] = 1
        one_hot_y = one_hot_y.T
        return one_hot_y

# validation function
def test_prediction(index, X, y, nn_model):
    current_image = X[:, index, None]
    prediction = nn_model.predict(X_train[:, index, None])
    label = y[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
    
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()



### Test on data
# Define dataset
data = pd.read_csv('./NeuralNetworkNumpy_train.csv').values
m, n = data.shape
# np.random.shuffle(data) # shuffle before splitting into dev and training sets

data_test = data[0:1000].T
y_test = data_test[0]
X_test = data_test[1:n]
X_test = X_test / 255.

data_train = data[1000:2000].T
y_train = data_train[0, :]
X_train = data_train[1:n, :]
X_train = X_train / 255.
_,m_train = X_train.shape

# Train our neural network!
network = NeuralNetworkNumpy()
network.train(X_train, y_train, n_iterations=1000, learning_rate=0.005)

# Make some predictions
test_prediction(0, X_test, y_test, network)
test_prediction(1, X_test, y_test, network)
test_prediction(3, X_test, y_test, network)
