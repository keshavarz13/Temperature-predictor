import numpy as np 

class GradientDescent:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def do_regression(self, X, y, iteration):
        self.w = np.array(np.random.rand(len(X[0]) + 1), dtype=np.float64)
        for i in range(iteration):
            self.learning_rate -= self.learning_rate * (i / (iteration * 1.1))
            for j in range(X.shape[0]):
                self.w += self.learning_rate * \
                    (y[j] - np.dot(np.append(np.ones(1), X[j]), self.w)) * \
                    np.append(np.ones(1), X[j])

    def predic_y(self, X):
        z = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            x_row = np.append(np.ones(1), X[i])
            z[i] = np.dot(x_row, self.w)
        return z