import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

# Initializing theta0 and theta1 values to 0
theta = np.zeros((2, 1))


def normalize_features(x: np.array) -> np.array:
    """
    Use the Standardization technique to make the training faster
    by having features on the same scale and prevent the ones
    with large values from dominating the learning process

    :param: x: features from data set, car mileage in our case
    :return: normalized x features from dataset
    """
    return (x - np.mean(x)) / np.std(x)


def denormalize_features(theta: np.array, x: np.array) -> tuple:
    """
    Denormalizes the theta parameters to their original scale

    :param theta: model parameters
    :param x: features from data set
    :return: denormalized theta parameters
    """
    theta0 = theta[0, 0] / np.std(x)
    theta1 = theta[1, 0] - (theta0 * np.mean(x))
    return (theta0, theta1)


def model(X: np.ndarray, theta: np.array) -> np.ndarray:
    """
    Calculates the predicted values based on the model parameters

    :param X: features from data set
    :param theta: model parameters
    :return: predicted values
    """
    return X.dot(theta)


def cost(X: np.ndarray, y: np.array, theta: np.array) -> float:
    """
    Calculates the error between predicted and expected values

    :param X: features from data set
    :param y: expected values from data set
    :param theta: model parameters
    :return: real number describing model performance
    """
    m = len(y)
    sum = 0

    for i in range(m):
        Y = model(X, theta)
        sum += (Y[i] - y[i])**2

    return sum / (2 * m)


def gradient(X: np.ndarray, y: np.array, theta: np.array) -> np.ndarray:
    """
    Calculates the gradient of the cost function
    according to theta model parameters

    :param X: features from data set
    :param y: expected values from data set
    :param theta: model parameters
    :return: gradient of the cost function
    """
    m = len(y)
    error = model(X, theta) - y

    theta0_gradient = (1 / m) * np.sum(error)
    theta1_gradient = (1 / m) * np.sum(error * X[:, 0].reshape(-1, 1))

    return np.array([[theta1_gradient], [theta0_gradient]])


def gradient_descent(
    X: np.ndarray,
    y: np.array,
    theta: np.array,
    learning_rate: float,
    n_iterations: int
) -> tuple:
    """
    Adjusts the model’s parameters to minimize errors
    finding the best-fit line using gradient descent algorithm

    :param X: features from data set
    :param y: expected values from data set
    :param theta: model parameters
    :param learning_rate: step size for each iteration
    :param n_iterations: number of iterations to perform
    :return: tuple containing the final theta, cost and theta history
    """
    cost_evolution = np.zeros(n_iterations)
    theta_history = [[0, 0] for _ in range(10)]

    index = 0
    for i in range(n_iterations):
        theta = theta - (gradient(X, y, theta) * learning_rate)

        cost_evolution[i] = cost(X, y, theta).item()
        if i % 50 == 0 and index < 10:
            theta_history[index] = theta
            index += 1

    return theta, cost_evolution, theta_history


def coef_determination(y: np.array, pred: np.ndarray) -> np.ndarray:
    """
    Calculates the coefficient of determination (R^2)
    to evaluate the model's performance

    :param y: expected values from data set
    :param pred: predicted values from the model
    :return: coefficient of determination (R^2)
    """
    u = 0
    v = 0
    m = y.size
    y_mean = y.mean()

    for i in range(m):
        u += (y[i] - pred[i])**2
        v += (y[i] - y_mean)**2

    return 1 - u / v


def mean_squared_error(y: np.array, pred: np.ndarray) -> np.ndarray:
    """
    Calculates the mean squared error (MSE)
    between expected and predicted values

    :param y: expected values from data set
    :param pred: predicted values from the model
    :return: mean squared error
    """
    u = 0
    m = y.size

    for i in range(m):
        u += (y[i] - pred[i])**2

    return u / m


def root_mean_squared_error(error: float) -> float:
    """
    Calculates the root mean squared error (RMSE)

    :param error: mean squared error
    :return: root mean squared error
    """
    return math.sqrt(error)


def mean_absolute_error(y: np.array, pred: np.ndarray) -> float:
    """
    Calculates the mean absolute error (MAE)
    between expected and predicted values

    :param y: expected values from data set
    :param pred: predicted values from the model
    :return: mean absolute error
    """
    u = 0
    m = y.size

    for i in range(m):
        u += abs(y[i][0] - pred[i][0])

    return u / m


if __name__ == "__main__":
    # Load the dataset and convert it to a DataFrame
    car_path = "./data.csv"
    df = pd.read_csv(car_path)

    # Convert mileage and price lists into 2D column vectors
    x = np.array(df["km"].values)
    y = np.array(df["price"].values)
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)

    x_normalized = normalize_features(x)
    X = np.hstack((x_normalized, np.ones(x_normalized.shape)))
    print(f'Normalized KM (features) Values: :\n{X}')

    # calculate cost before training
    print(f'Initial cost: {cost(X, y, theta)}')
    learning_rate = 0.01
    n_iterations = 500
    print(theta)

    # train model
    theta, cost_evolution, theta_history = gradient_descent(
        X, y, theta, learning_rate, n_iterations
    )
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

    # Denormalize theta history
    y_history = [None] * 10
    for i in range(10):
        theta_history[i] = denormalize_features(theta_history[i], x)
        y_history[i] = model(X_raw, theta_history[i])

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
    print(f"The absolute error in the prediction of the model (MAE) is {absolute_error:.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left Subplot: Manual Gradient Descent Regression
    axes[0].plot(np.arange(n_iterations), cost_evolution, color='pink', label='Cost Evolution')
    axes[0].set_title('Cost Evolution')
    axes[0].set_xlabel('Iterations')
    axes[0].set_ylabel('Cost')
    axes[0].legend()

    # Right Subplot: Polyfit Regression
    axes[1].scatter(x, y, color='blue', label='Data')
    for i in range(10):
        plt.plot(x, y_history[i], label=f'Regression {i * 50}', linestyle=':')

    axes[1].set_title('Linear Regression Evolution')
    axes[1].set_xlabel('Mileage')
    axes[1].set_ylabel('Price')
    axes[1].legend()

    plt.tight_layout()
    plt.show()

    # plot polyfit and trained model and compare
    theta_polyfit = np.polyfit(x.flatten(), y, 1)
    prediction_polyfit = theta_polyfit[0] * x + theta_polyfit[1]

    # Scatter plot of the dataset
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left Subplot: Manual Linear Regression
    axes[0].scatter(x, y, color='blue', label='Data')
    axes[0].plot(x, y_pred, color='red', label='Manual Regression')
    axes[0].set_title('Manual Linear Regression')
    axes[0].set_xlabel('Mileage')
    axes[0].set_ylabel('Price')
    axes[0].legend()

    # Right Subplot: Polyfit Regression
    axes[1].scatter(x, y, color='blue', label='Data')
    axes[1].plot(x, prediction_polyfit, color='green', label='Polyfit Regression')
    axes[1].set_title('Polyfit Linear Regression')
    axes[1].set_xlabel('Mileage')
    axes[1].set_ylabel('Price')
    axes[1].legend()

    plt.tight_layout()
    plt.show()

