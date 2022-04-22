#   Neural network with:
#   - a input layer with 2 features (x1, x2)
#   - a hidden layer with 2 neurons (h1, h2) with Relu activation
#   - an output layer with 1 neuron (o1) with sigmoid activation
#   - activation function: sigmoid function (can also be ReLU)
#   - output layer function: sigmoid function (can also be SoftMax)
#   - loss function: MSE (can also be cross-entropy)

# Steps when training a neural network:
# 1. Randomly initialize weights (theta)
# 2. Implement forward propagation to get y_hat_i for any x_i
# 3. Implement code to compute cost function J(theta)
# 4. Implement backprop to compute partial derivatives d_J_d_theta
# 5. Use gradient descent to update all weights (theta) and minimize J(theta)

# for i = 1:m:
# perform forward prop and back prop using (x(i), y(i))
# (Get activations a(l) and delta terms delta(l) for l=2, ..., L)

# Important functions
# Error at layer l:
# delta(l) := delta(l) + delta(l+1)*(a(l))T

# Reference illustration and video:
# https://www.kaggle.com/code/wwsalmon/simple-mnist-nn-from-scratch-numpy-no-tf-keras/notebook
# https://www.youtube.com/watch?v=w8yWXqWQYmU
# https://www.adeveloperdiary.com/data-science/deep-learning/neural-network-with-softmax-in-python/

import numpy as np

class NeuralNetwork:
    def __init__(self):
        """
        X: [[x11, x12], ...[xm1, xm2]] - for simplicity, assume x has two features
        y: [y1, y2, ..., ym]; y = 0 or 1
        """
        self.X = None
        self.y = None
        self.m = None
        self.n = None

        ### weights and biases
        # hidden layer, neuron h1
        self.w1 = None # e.g. w1 = [0.1, 0.2]
        self.w2 = None
        self.b1 = None
        # hidden layer, neuron h2
        self.w3 = None # e.g. w3 = [0.3, 0.1]
        self.w4 = None
        self.b2 = None
        # output layer, neuron o1
        self.w5 = None # e.g. w5 = [0.4, 0.3]
        self.w6 = None 
        self.b3 = None

        ### derivatives of loss vs. weights/biases
        self.d_L_d_w1 = None
        self.d_L_d_w2 = None
        self.d_L_d_b1 = None
        self.d_L_d_w3 = None
        self.d_L_d_w4 = None
        self.d_L_d_b2 = None
        self.d_L_d_w5 = None
        self.d_L_d_w6 = None
        self.d_L_d_b3 = None

    def train(self, X, y, n_iterations=1000, learning_rate=0.05):
        m = len(X)
        self.initialize_params()
        for _ in range(n_iterations):
            y_pred = self.forward_prop(X)
            self.back_prop(X, y, y_pred, m)
            self.update_params(learning_rate)
        return None

    def predict(self, x):
        return self.forward_prop([x])[0]

    def initialize_params(self):
        self.w1 = np.random.normal()
        self.w2 = np.random.normal()
        self.b1 = np.random.normal()
        self.w3 = np.random.normal()
        self.w4 = np.random.normal()
        self.b2 = np.random.normal()
        self.w5 = np.random.normal()
        self.w6 = np.random.normal()
        self.b3 = np.random.normal()
        
    def forward_prop(self, X):
        y_pred = []
        for x in X:
            # hidden layer 1
            z1 = self.w1*x[0] + self.w2*x[1] + self.b1
            h1 = self.relu(z1)
            z2 = self.w3*x[0] + self.w4*x[1] + self.b2
            h2 = self.relu(z2)
            # output layer
            z3 = self.w5*h1 + self.w6*h2 + self.b3
            o1 = self.sigmoid(z3)
            y_pred.append(o1)
        return y_pred

    def back_prop(self, X, y, y_pred, m):
        self.d_L_d_w1 = 0
        self.d_L_d_w2 = 0
        self.d_L_d_b1 = 0
        self.d_L_d_w3 = 0
        self.d_L_d_w4 = 0
        self.d_L_d_b2 = 0
        self.d_L_d_w5 = 0
        self.d_L_d_w6 = 0
        self.d_L_d_b3 = 0
        for i in range(m):
            x_i = X[i]
            y_i = y[i]
            y_pred_i = y_pred[i]

            ### intermediate values during forward propagation
            # hidden layer 1
            z1 = self.w1*x_i[0] + self.w2*x_i[1] + self.b1
            h1 = self.relu(z1)
            z2 = self.w3*x_i[0] + self.w4*x_i[1] + self.b2
            h2 = self.relu(z2)
            # output layer
            z3 = self.w5*h1 + self.w6*h2 + self.b3
            o1 = self.sigmoid(z3)
            
            ### get gradients for each weight/bias
            # output layer
            d_L_d_o1 = y_pred_i - y_i
            d_o1_d_z3 = self.sigmoid_deriv(z3)
            d_L_d_z3 = d_L_d_o1 * d_o1_d_z3
            self.d_L_d_w5 += 1/m * d_L_d_z3 * h1
            self.d_L_d_w6 += 1/m * d_L_d_z3 * h2
            self.d_L_d_b3 += 1/m * d_L_d_z3
            ### hidden layer
            # hidden neuron h1
            d_L_d_z1 = d_L_d_z3 * self.w5 * self.relu_deriv(z1)
            self.d_L_d_w1 += 1/m * d_L_d_z1 * x_i[0]
            self.d_L_d_w2 += 1/m * d_L_d_z1 * x_i[1]
            self.d_L_d_b1 += 1/m * d_L_d_z1
            # hidden neuron h2
            d_L_d_z2 = d_L_d_z3 * self.w6 * self.relu_deriv(z2)
            self.d_L_d_w3 += 1/m * d_L_d_z2 * x_i[0]
            self.d_L_d_w4 += 1/m * d_L_d_z2 * x_i[1]
            self.d_L_d_b2 += 1/m * d_L_d_z2

    def update_params(self, learning_rate):
        self.w1 -= learning_rate * self.d_L_d_w1
        self.w2 -= learning_rate * self.d_L_d_w2
        self.b1 -= learning_rate * self.d_L_d_b1
        self.w3 -= learning_rate * self.d_L_d_w3
        self.w4 -= learning_rate * self.d_L_d_w4
        self.b2 -= learning_rate * self.d_L_d_b2
        self.w5 -= learning_rate * self.d_L_d_w5
        self.w6 -= learning_rate * self.d_L_d_w6
        self.b3 -= learning_rate * self.d_L_d_b3

    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))

    def sigmoid_deriv(self, x):
        fx = self.sigmoid(x)
        return fx * (1 - fx)
    
    def relu(self, x):
        return np.max(x, 0)
    
    def relu_deriv(self, x):
        return x > 0


# Define dataset
X = [
  [-2, -1],  # Alice
  [25, 6],   # Bob
  [17, 4],   # Charlie
  [-15, -6], # Diana
]
y = [
  1, # Alice
  0, # Bob
  0, # Charlie
  1, # Diana
]

# Train our neural network!
network = NeuralNetwork()
network.train(X, y)

# Make some predictions
emily = [-7, -3] # 128 pounds, 63 inches
frank = [20, 2]  # 155 pounds, 68 inches
print("Emily: %.3f" % network.predict(emily)) # 0.951 - F
print("Frank: %.3f" % network.predict(frank)) # 0.039 - M