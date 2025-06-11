import os
import sys
import numpy as np
import pandas as pd


def predict(mileage: int, theta: np.array) -> np.array:
    return theta[1] + (theta[0] * mileage)


if __name__ == "__main__":
    try:
        if len(sys.argv) < 2:
            print('Error: You must provide a car mileage.')
        elif int(sys.argv[1]) < 0:
            print('Error: Car mileage must be a positive value.')
        else:
            # Load your dataset
            df = pd.read_csv('./data.csv')
            x = df['km'].values
            y = df['price'].values

            if os.path.exists('theta.npy'):
                theta = np.load('theta.npy')
            else:
                theta = np.zeros((2, 1))

            predicted_price = predict(int(sys.argv[1]), theta)
            print(predicted_price[0])
    except ValueError:
        print('Error: Invalid input. Please provide a valid car mileage as a positive integer.')
