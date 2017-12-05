import numpy as np
import sys, time
import math
#import matplotlib.pyplot as plt
import operator
import pandas as pd



def read_laplace_data():
    df = pd.read_csv('laplace_data_225_2.csv')
    data = df.as_matrix().astype(np.float64)
    return data[:, 1:-1], data[:, -1]


def read_normalized_laplace_data():
    X, Y = read_laplace_data()
    print(X[1,:])
    X = X+2
    X = np.log(X)

    col = X.shape[1]
    print("col size is", col)
    normalize_data = []
    for c in range(col):
        Xl = X[:, c]
        aver = (max(Xl) + min(Xl)) / 2
        deno = max(Xl) - min(Xl)
        X[:, c] = X[:, c] - aver
        X[:, c] = X[:, c] / deno
        normalize_data.append([aver, deno]) 
    print("\n\n\n\n")

    return X, Y, normalize_data

    

if __name__ == '__main__':
    main()