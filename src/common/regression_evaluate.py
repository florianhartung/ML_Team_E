import numpy as np

def mse(predict, actual):
    return np.mean((predict - actual) ** 2)

def r2(predict, actual):
    return 1 - np.sum((predict - actual) ** 2) / np.sum((actual - np.mean(actual)) ** 2)