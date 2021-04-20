"""
    TEST SCRIPT FOR POLYNOMIAL REGRESSION 1
    AUTHOR Eric Eaton, Xiaoxiang Hu
"""

import numpy as np
import matplotlib.pyplot as plt
from polyreg import PolynomialRegression

single_plot = True

def plotUnivariate(X, y, d, reg_lambda):

    model = PolynomialRegression(degree=d, reg_lambda=reg_lambda)
    model.fit(X, y)

    # output predictions
    xpoints = np.linspace(np.max(X), np.min(X), 100).reshape(-1, 1)
    ypoints = model.predict(xpoints)

    # plot curve
    plt.plot(X, y, 'rx')
    plt.title('PolyRegression with d = ' + str(d) + ', lambda = ' + str(reg_lambda))
    plt.plot(xpoints, ypoints, 'b-')
    plt.xlabel('X')
    plt.ylabel('Y')

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
    reg_lambda = 0

    if single_plot:
        plotUnivariate(X, y, d, reg_lambda)
        plt.show()

        # show isn't working for me for some reason
        plt.savefig("a4_" + str(reg_lambda) + ".png")

    else:

        reg_lambda_list = [0, 1E-8, 1E-6, 1E-4, 1E-2, 1E-1, 1, 100]
        plt.figure(figsize=(20, 10), dpi=100)

        for idx, reg_lambda in enumerate(reg_lambda_list):
            plt.subplot(2, 4, idx + 1)
            plotUnivariate(X, y, d, reg_lambda)
        
        plt.savefig("a4_compilation.png")
