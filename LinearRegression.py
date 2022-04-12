## Linear Regression
# Assumption: Relationship between X and y is linear
# Equation: y = beta0 + beta1*x
# Error: MSE = sum(y-yhat)^2/N
# Gradient Descent: Reduce the error over multiple iterations
#   1. Start with a random guess of betas
#   2. Compute the MSE
#   3. Compute the gradients and updates betas
#   4. Repeat until convergence

class LinearRegression:
    def __init__(self):
        self.X = None
        self.y = None

    def linear_regression(self, X, y, iterations=100, learning_rate=0.1):
        m, n = len(X), len(X[0])
        beta_0, beta_other = self.initialize_params(n)
        for _ in range(iterations):
            gradient_beta_0, gradient_beta_other = self.compute_gradient(X, y, beta_0, beta_other, m, n)
            beta_0, beta_other = update_params(beta_0, beta_other, gradient_beta_0, gradient_beta_other, learning_rate)
        return beta_0, beta_other

    
    def predict(self, x_new):
        pass