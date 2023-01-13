import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import SingleTargetRegression as STR
import MultiTargetRegression as MTR
import KNeighborsRegression as KNR


def apply_degree(X, degree):
    result = []
    for x in X:
        row = []
        for d in range(degree):
            foo = (list(np.power(x, d+1)))
            row += foo
        result.append(row)
    return np.array(result)


def feature_normalize(X):
    return ((X - X.mean()) / (X.max() - X.min()))


def plot_prediction(title, test_input, real_values, predicted_values):
    plt.ylabel(title)
    plt.plot(range(test_input.shape[0]), real_values, label="real", color='y')
    plt.plot(range(test_input.shape[0]), predicted_values, label="prediction", color='g')
    plt.legend()
    plt.show()


def read_and_prepare_data(fileAddress="./dataset.csv"):
    data = pd.read_csv(fileAddress)
    selected_features = data.columns[2:][:-2]
    data = data.dropna()
    data[selected_features] = feature_normalize(data[selected_features])

    split_date = '2016-01-01'
    train_data = data.loc[data['Date'] <= split_date]
    test_data = data.loc[data['Date'] > split_date]

    training_next_min = train_data['Next_Tmin']
    training_next_max = train_data['Next_Tmax']
    train_data = train_data[selected_features]

    testing_next_min = test_data['Next_Tmin']
    testing_next_max = test_data['Next_Tmax']
    test_data = test_data[selected_features]

    return train_data.to_numpy(), training_next_min.to_numpy(), training_next_max.to_numpy(), test_data.to_numpy(), testing_next_min.to_numpy(), testing_next_max.to_numpy()


# Main -----------------------------------------------------------------------------
# part a
train_input, training_next_min, training_next_max, test_input, testing_next_min, testing_next_max = read_and_prepare_data()

# part b
train_input = apply_degree(train_input, 5)
test_input = apply_degree(test_input, 5)
regression = STR.GradientDescent(learning_rate=0.01)
regression.do_regression(train_input, training_next_max, 800)
predicted_y = regression.predic_y(test_input)
mse = metrics.mean_squared_error(testing_next_max, predicted_y)
print("SSE =", mse * testing_next_max.shape[0])
print("MSE =", mse)
plot_prediction("Next Tmax", test_input, testing_next_max, predicted_y)


# part c-1
regression = MTR.GradientDescent(learning_rate=0.0001)
regression.do_regression(train_input, np.concatenate(
    (training_next_max.reshape(-1, 1), training_next_min.reshape(-1, 1)), axis=1), 10000)
predicted_y = regression.predic_y(test_input)

mse = metrics.mean_squared_error(testing_next_max, predicted_y[:, 0])
print("T_max")
print("SSE:", mse * testing_next_max.shape[0])
print("MSE:", mse)
plot_prediction("Next Tmax (Single Target Regression)", test_input, testing_next_max, predicted_y[:, 0])


mse = metrics.mean_squared_error(testing_next_min, predicted_y[:, 1])
print("T_min")
print("SSE:", mse * testing_next_max.shape[0])
print("MSE:", mse)
plot_prediction("Next Tmin", test_input, testing_next_min, predicted_y[:, 1])

# part c-2
regression = MTR.NormalEquation()
regression.do_regression(train_input, np.concatenate(
    (training_next_max.reshape(-1, 1), training_next_min.reshape(-1, 1)), axis=1))
predicted_y = regression.predic_y(test_input)

mse = metrics.mean_squared_error(testing_next_max, predicted_y[:, 0])
print("T_max")
print("SSE:", mse * testing_next_max.shape[0])
print("MSE:", mse)
plot_prediction("Next Tmax", test_input, testing_next_max, predicted_y[:, 0])


mse = metrics.mean_squared_error(testing_next_min, predicted_y[:, 1])
print("T_min")
print("SSE:", mse * testing_next_max.shape[0])
print("MSE:", mse)
plot_prediction("Next Tmin", test_input, testing_next_min, predicted_y[:, 1])

# part d
regression = KNR.KNeighborsRegression(k=5)
predicted_y = regression.predict_y(train_input, np.concatenate(
    (training_next_max.reshape(-1, 1), training_next_min.reshape(-1, 1)), axis=1), test_input)

mse = metrics.mean_squared_error(testing_next_max, predicted_y[:, 0])
print("T_max")
print("SSE:", mse * testing_next_max.shape[0])
print("MSE:", mse)
plot_prediction("Next Tmax", test_input, testing_next_max, predicted_y[:, 0])

mse = metrics.mean_squared_error(testing_next_min, predicted_y[:, 1])
print("T_min")
print("SSE:", mse * testing_next_max.shape[0])
print("MSE:", mse)
plot_prediction("Next Tmin", test_input, testing_next_min, predicted_y[:, 1])