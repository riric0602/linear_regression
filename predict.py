import sys
import numpy as np
import pandas as pd


def predict(mileage: int, theta: np.array) -> int:
    return theta[0] + (theta[1] * mileage)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Error: You must provide a car mileage.')
    elif int(sys.argv[1]) < 0:
        print('Error: Car mileage must be a positive value.')
    else:
        # Load your dataset
        df = pd.read_csv('./data.csv')
        x = df['km'].values
        y = df['price'].values

        theta = np.load('theta.npy')
        print(predict(int(sys.argv[1]), theta))
