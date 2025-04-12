import sys

def predict(mileage: int) -> int:
    return theta[1] + (theta[0] * mileage)

if __name__ == "__main__":
    print(predict(int(sys.argv[1])))