from tqdm import tqdm
import numpy as np

class KNeighborsRegression:
    def __init__(self, k):
        self.k = k

    def predict_y(self, X_train, y_train, X_test):
        predictions = []
        for i, test_instance in enumerate(X_test):
            distances = []
            for i, train_instance in enumerate(X_train):
                distances.append((np.linalg.norm(test_instance - train_instance), i))
            predictions.append(sum([y_train[i] for _, i in sorted(distances)[:self.k]]) / self.k)
        return np.array(predictions)