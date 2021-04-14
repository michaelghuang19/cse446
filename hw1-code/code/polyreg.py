'''
    Template for polynomial regression
    AUTHOR Eric Eaton, Xiaoxiang Hu
'''

import numpy as np


#-----------------------------------------------------------------
#  Class PolynomialRegression
#-----------------------------------------------------------------

class PolynomialRegression:

    def __init__(self, degree=1, reg_lambda=1E-8):
        """
        Constructor
        """

        self.degree = degree
        self.reg_lambda = reg_lambda

        self.mean_list = None
        self.std_list = None
        self.theta = None

    def polyfeatures(self, X, degree):
        """
        Expands the given X into an n * d array of polynomial features of
            degree d.

        Returns:
            A n-by-d numpy array, with each row comprising of
            X, X * X, X ** 3, ... up to the dth power of X.
            Note that the returned matrix will not include the zero-th power.

        Arguments:
            X is an n-by-1 column numpy array
            degree is a positive integer
        """
        
        n = len(X)
        d = degree

        result = np.zeros((n, d))

        for i in range(n):
            for j in range(d):
                result[i][j - 1] = X[i] ** (j + 1)

        return result

    def standardize(self, matrix):
        # TODO: check if this is behaving properly
        n, d = matrix.shape

        result = np.zeros((n, d))
        for i in range(d):
            result[:, i] = (matrix[:, i] - self.mean_list[i]) / self.std_list[i]

        return result


    def fit(self, X, y):
        """
            Trains the model
            Arguments:
                X is a n-by-1 array
                y is an n-by-1 array
            Returns:
                No return value
            Note:
                You need to apply polynomial expansion and scaling
                at first
        """
        #TODO

        n = len(X)
        d = self.degree

        poly_matrix = self.polyfeatures(X, d)
        
        # get the means/stds of training data
        self.mean_list = np.zeros(d)
        self.std_list = np.zeros(d)

        for i in range(d):
            cur = poly_matrix[:, i]
            cur = cur.tolist()

            self.mean_list[i] = np.mean(cur)
            self.std_list[i] = np.std(cur)

        poly_matrix = self.standardize(poly_matrix)

        # add the x0 column of 1s
        np.concatenate((np.ones((n, 1)), poly_matrix), axis=1)

        # construct reg matrix
        reg_matrix = self.reg_lambda * np.eye(d)
        reg_matrix[0, 0] = 0

        # Since 0-mean, we can do 
        # weight = (X^T*X)^-1 X^T*Y
        self.theta = np.linalg.pinv(poly_matrix.T.dot(poly_matrix)
                    + reg_matrix).dot(poly_matrix.T).dot(y)

    def predict(self, X):
        """
        Use the trained model to predict values for each instance in X
        Arguments:
            X is a n-by-1 numpy array
        Returns:
            an n-by-1 numpy array of the predictions
        """
        # TODO

        n = len(X)
        d = self.degree

        poly_matrix = self.polyfeatures(X, d)
        poly_matrix = self.standardize(poly_matrix)

        # add column of 1s at beginning
        np.concatenate((np.ones((n, 1)), poly_matrix), axis=1)

        # return predictions
        return poly_matrix.dot(self.theta)

#-----------------------------------------------------------------
#  End of Class PolynomialRegression
#-----------------------------------------------------------------


def learningCurve(Xtrain, Ytrain, Xtest, Ytest, reg_lambda, degree):
    """
    Compute learning curve

    Arguments:
        Xtrain -- Training X, n-by-1 matrix
        Ytrain -- Training y, n-by-1 matrix
        Xtest -- Testing X, m-by-1 matrix
        Ytest -- Testing Y, m-by-1 matrix
        reg_lambda -- regularization factor
        degree -- polynomial degree

    Returns:
        errorTrain -- errorTrain[i] is the training accuracy using
        model trained by Xtrain[0:(i+1)]
        errorTest -- errorTrain[i] is the testing accuracy using
        model trained by Xtrain[0:(i+1)]

    Note:
        errorTrain[0:1] and errorTest[0:1] won't actually matter, since we start displaying the learning curve at n = 2 (or higher)
    """

    n = len(Xtrain)

    errorTrain = np.zeros(n)
    errorTest = np.zeros(n)

    #TODO -- complete rest of method; errorTrain and errorTest are already the correct shape

    return errorTrain, errorTest
