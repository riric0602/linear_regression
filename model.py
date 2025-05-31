import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

# Initializing theta0 and theta1 values to 0
theta = np.zeros((2, 1))


def normalize_features(x: np.ndarray) -> np.ndarray:
    """
    Use the Standardization technique to make the training faster
    by having features on the same scale, to prevent the ones
    with large values from dominating the learning process

    :param: x: features from data set, car mileage in our case
    :return: normalized x features from dataset
    """
    return (x - np.mean(x)) / np.std(x)


def denormalize_features(theta, x):
    theta0 = theta[0, 0] / np.std(x)
    theta1 = theta[1, 0] - (theta0 * np.mean(x))
    return (theta0, theta1)


def model(X, theta):
    return X.dot(theta)


def cost(X, y, theta):
    m = len(y)
    sum = 0

    for i in range(m):
        Y = model(X, theta)
        sum += (Y[i] - y[i])**2

    return sum / (2 * m)


def gradient(X, y, theta):
    m = len(y)
    error = model(X, theta) - y

    theta1_gradient = (1 / m) * np.sum(error * X[:, 0].reshape(-1, 1))
    theta0_gradient = (1 / m) * np.sum(error)

    return np.array([[theta1_gradient], [theta0_gradient]])


def gradient_descent(X, y, theta, learning_rate, n_iterations):
    for i in range(n_iterations):
        theta = theta - (gradient(X, y, theta) * learning_rate)
    return theta


def coef_determination(y, pred):
    u = 0
    v = 0
    m = y.size
    y_mean = y.mean()

    for i in range(m):
        u += (y[i] - pred[i])**2
        v += (y[i] - y_mean)**2

    return 1 - u / v


def mean_squared_error(y, pred):
    u = 0
    m = y.size

    for i in range(m):
        u += (y[i] - pred[i])**2

    return u / m


def root_mean_squared_error(error):
    return math.sqrt(error)


def mean_absolute_error(y, pred):
    u = 0
    m = y.size

    for i in range(m):
        u += abs(y[i] - pred[i])

    return u / m


if __name__ == "__main__":
    car_path = "./data.csv"
    df = pd.read_csv(car_path)
    print(f'DataFrame: \n{df}', flush=True)

    x = df["km"].values
    y = df["price"].values
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)

    x_normalized = normalize_features(x)
    print(f'Normalized KM (features) Values: \n{x_normalized}')

    X = np.hstack((x_normalized, np.ones(x_normalized.shape)))
    print(f'X :\n{X}')

    # calculate cost before and after training
    print(f'Initial cost: {cost(X, y, theta)}')
    learning_rate = 0.01
    n_iterations = 500
    print(theta)

    # train model
    theta = gradient_descent(X, y, theta, learning_rate, n_iterations)
    print(f'Trained cost: {cost(X, y, theta)}')

    # Rebuild theta into proper shape for model()
    theta = denormalize_features(theta, x)
    print(f'Updated denormalized theta: {theta}')
    theta = np.array(theta).reshape(-1, 1)

    # Save theta
    theta_file = 'theta.npy'
    np.save(theta_file, theta)

    # Create X like before but with raw x
    X_raw = np.hstack((x, np.ones(x.shape)))
    y_pred = model(X_raw, theta)

    # Scatter plot of the dataset
    plt.scatter(x, y, color='blue', label='Data')
    # Regression line
    plt.plot(x, y_pred, color='red', label='Manual Regression')
    plt.title('Manual Linear Regression')
    plt.xlabel('Mileage')
    plt.ylabel('Price')
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    plt.show()

    # make statistics
    coef = coef_determination(y, y_pred) * 100
    print(f"The precision of the model (R^2) is {coef[0]:.2f}%")

    error = mean_squared_error(y, y_pred)
    print(f"The average error in the prediction of the model (MSE) is {error[0]:.2f}")

    root_error = root_mean_squared_error(error[0])
    print(f"The root average error in the prediction of the model (RMSE) is {root_error:.2f}")

    absolute_error = mean_absolute_error(y, y_pred)
    print(f"The absolute error in the prediction of the model (MAE) is {absolute_error[0]:.2f}")

    # plot evolution of cost
    
    
    # plot polyfit and trained model and compare

