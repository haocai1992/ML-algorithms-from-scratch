# Logistic Regression:
# binary classification
# z = theta_0 + theta_other * X
# y_hat = 1 / (1 + e^-z)
# cost function J(theta_0, theta_other) = -1/m sum(i=1 to m) (yi * log(y_hat_i) + (1-yi) * log(1-y_hat_i))
# gradient descent process:
# theta_0 += learning_rate * sum(i=1 to m) (y_i - h_theta(xi))
# theta_other += learning_rate * sum(i=1 to m) (y_i - h_theta(xi)) * xi

import random
import math

class LogisticRegression:
    def __init__(self):
        """
        X: [[x11, x12, ..., x1n], ...[xm1, xm2, ..., xmn]]
        y: [y1, y2, ..., ym]; y = 0 or 1
        """
        self.X = None
        self.y = None
        self.m = None
        self.n = None
        self.theta_0 = None
        self.theta_other = None

    def train(self, X, y, n_iterations, learning_rate):
        m, n = len(X), len(X[0])
        theta_0, theta_other = self.initialize_params(n=n) # Time Complexity: O(N)
        for _ in range(n_iterations): # Time Complexity: O(I*M*N)
            gradient_theta_0, gradient_theta_other = self.compute_gradients(X, y, theta_0, theta_other, m, n)
            theta_0, theta_other = self.update_params(theta_0, theta_other, gradient_theta_0, gradient_theta_other, learning_rate)
        return theta_0, theta_other

    def predict(self, x_new, theta_0, theta_other):
        return self.sigmoid(x_new, theta_0, theta_other)

    def initialize_params(self, n):
        theta_0 = random.random()
        theta_other = [random.random() for _ in range(n)]
        return theta_0, theta_other

    def compute_gradients(self, X, y, theta_0, theta_other, m, n):
        gradient_theta_0 = 0
        gradient_theta_other = [0 for j in range(n)] # Space complexity: O(N)
        for i in range(m):
            x = X[i]
            y_hat = self.sigmoid(x, theta_0, theta_other)
            dJ_dtheta0 = y[i] - y_hat
            gradient_theta_0 += dJ_dtheta0 / m
            for j in range(n):
                dJ_dtheta_other_j = (y[i] - y_hat)*x[j]
                gradient_theta_other[j] += dJ_dtheta_other_j / m
        return gradient_theta_0, gradient_theta_other

    def compute_gradients_minibatch(self, X, y, theta_0, theta_other, m, n, batch_size):
        gradient_theta_0 = 0
        gradient_theta_other = [0 for j in range(n)] # Space complexity: O(N)
        for _ in range(batch_size):
            i = random.randint(0, m-1)
            x = X[i]
            y_hat = self.sigmoid(x, theta_0, theta_other)
            dJ_dtheta0 = y[i] - y_hat
            gradient_theta_0 += dJ_dtheta0 / m
            for j in range(n):
                dJ_dtheta_other_j = (y[i] - y_hat)*x[i][j]
                gradient_theta_other[j] += dJ_dtheta_other_j / m
        return gradient_theta_0, gradient_theta_other

    def sigmoid(self, x, theta_0, theta_other):
        return 1/(1 + math.exp(-(theta_0 + sum([theta_other[j] * x[j] for j in range(len(theta_other))]))))

    def update_params(self, theta_0, theta_other, gradient_theta_0, gradient_theta_other, learning_rate):
        theta_0 += learning_rate * gradient_theta_0
        for j in range(len(theta_other)):
            theta_other[j] += learning_rate * gradient_theta_other[j]
        return theta_0, theta_other


X = [[0, 1], [0, 2], [2, 1], [2, 4], [3, 7]]
y = [0, 0, 0, 1, 1]
x_new = [10, 12]

log_reg = LogisticRegression()
beta_0, beta_others = log_reg.train(X, y, n_iterations=100000, learning_rate=0.05)
print(beta_0, beta_others)
y_new = log_reg.predict(x_new, beta_0, beta_others)
print(y_new)