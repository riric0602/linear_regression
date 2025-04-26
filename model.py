import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

theta = np.zeros((2, 1))

def normalize_features(x: np.ndarray) -> np.ndarray:
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

    # calculate cost before training
    print(f'Initial cost: {cost(X, y, theta)}')
    learning_rate = 0.01
    n_iterations = 500
    print(theta)
    theta = gradient_descent(X, y, theta, learning_rate, n_iterations)
    # train model

    print(f'Trained cost: {cost(X, y, theta)}')

    theta = denormalize_features(theta, x)
    print(f'Updated denormalized theta: {theta}')

    # Rebuild theta into proper shape for model()
    theta = np.array(theta).reshape(-1, 1)

    # Save theta
    theta_file = 'theta.npy'
    np.save(theta_file, theta)

    # Now predict using your model function
    X_raw = np.hstack((x, np.ones(x.shape)))  # Create X like before but with raw x
    y_pred = model(X_raw, theta)

    plt.scatter(x, y, color='blue', label='Data')  # Scatter plot of the dataset
    plt.plot(x, y_pred, color='red', label='Manual Regression')  # Regression line
    plt.title('Manual Linear Regression')
    plt.xlabel('Mileage')
    plt.ylabel('Price')
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    plt.show()

    # plot linear regression
    # calculate new cost
    # compare cost and plot evolution of minimization algorithm
    # plot polyfit and trained model and compare
    # make statistics
