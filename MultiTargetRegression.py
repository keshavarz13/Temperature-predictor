import numpy as np

class GradientDescent :
    def __init__(self, learning_rate=0.0001):
        self.learning_rate = learning_rate

    def do_regression(self, X, y, iterations=10000):
        y = y / 50
        X = np.column_stack([np.ones(X.shape[0]), X])
        self.weights = np.random.rand(X.shape[1], y.shape[1]) - 0.5
        for _ in range(iterations):
            self.weights -= self.learning_rate * \
                (X.T @ ((X @ self.weights) - y))

    def predic_y(self, X_test):
        return (np.column_stack([np.ones(X_test.shape[0]), X_test])) @ self.weights * 50


class NormalEquation :
    def __init__(self):
        self.weights = []

    def do_regression(self, X, y):
        X = np.column_stack([np.ones(X.shape[0]), X])
        self.weights = np.linalg.inv(X.T @ X) @ X.T @ y

    def predic_y(self, X_test):
        X_test = np.column_stack([np.ones(X_test.shape[0]), X_test])
        return X_test @ self.weights