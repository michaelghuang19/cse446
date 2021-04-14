"""
    TEST SCRIPT FOR POLYNOMIAL REGRESSION 1
    AUTHOR Eric Eaton, Xiaoxiang Hu
"""

import numpy as np
import matplotlib.pyplot as plt
from polyreg import PolynomialRegression

if __name__ == "__main__":
    '''
        Main function to test polynomial regression
    '''

    # load the data
    filePath = "../data/polydata.dat"
    file = open(filePath,'r')
    allData = np.loadtxt(file, delimiter=',')

    X = allData[:, [0]]
    y = allData[:, [1]]

    # regression with degree = d
    d = 8
    reg_lambda = 1E-1
    model = PolynomialRegression(degree=d, reg_lambda=reg_lambda)
    model.fit(X, y)

    # output predictions
    xpoints = np.linspace(np.max(X), np.min(X), 100).reshape(-1, 1)
    ypoints = model.predict(xpoints)

    # plot curve
    plt.figure()
    plt.plot(X, y, 'rx')
    plt.title('PolyRegression with d = '+str(d))
    plt.plot(xpoints, ypoints, 'b-')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

    # show isn't working for me for some reason
    plt.savefig("a4_" + str(reg_lambda) + ".png")
