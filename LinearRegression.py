## Linear Regression
# Assumption: Relationship between X and y is linear
# Equation: y = beta0 + beta1*x
# Error: MSE = sum(y-yhat)^2/N
# Gradient Descent: Reduce the error over multiple iterations
#   1. Start with a random guess of betas
#   2. Compute the MSE
#   3. Compute the gradients and updates betas
#   4. Repeat until convergence
import random

class LinearRegression:
    def __init__(self):
        """
        X = [[x01, ..., x0n], [x11, ..., x1n], ... [xm1, ..., xmn]]
        y = [y0, y1, ..., ym]
        """
        self.X = None
        self.y = None

    def linear_regression(self, X, y, iterations=100, learning_rate=0.1):
        m, n = len(X), len(X[0])
        beta_0, beta_other = self.initialize_params(n)
        for _ in range(iterations): # Time complexity: O(I * m * n)
            gradient_beta_0, gradient_beta_other = self.compute_gradient(X, y, beta_0, beta_other, m, n)
            beta_0, beta_other = self.update_params(beta_0, beta_other, gradient_beta_0, gradient_beta_other, learning_rate)
        return beta_0, beta_other

    
    def predict(self, x_new, beta_0, beta_others):
        return beta_0 + sum([beta_others[j]* x_new[j] for j in range(len(beta_others))])

    
    def initialize_params(self, n):
        beta_0 = 0
        beta_other = [random.random() for _ in range(n)]
        return beta_0, beta_other

    
    def compute_gradient(self, X, y, beta_0, beta_other, m, n):
        gradient_beta_0 = 0
        gradient_beta_other = [0] * n

        for i in range(m):
            y_i_hat = beta_0 + sum([beta_other[j]*X[i][j] for j in range(n)])
            derror_dy = 2 * (y[i] - y_i_hat)
            for j in range(n):
                gradient_beta_other[j] += derror_dy * X[i][j] / m
            gradient_beta_0 += derror_dy / m
        return gradient_beta_0, gradient_beta_other

    
    def update_params(self, beta_0, beta_other, gradient_beta_0, gradient_beta_other, learning_rate):
        beta_0 += gradient_beta_0 * learning_rate
        for j in range(len(beta_other)):
            beta_other[j] += gradient_beta_other[j] * learning_rate
        return beta_0, beta_other


X = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]]
y = [2, 5, 8, 11, 14]
x_new = [10, 12]

lr = LinearRegression()
beta_0, beta_others = lr.linear_regression(X, y, iterations=100000, learning_rate=0.05)
print(beta_0, beta_others)
y_new = lr.predict(x_new, beta_0, beta_others)
print(y_new)