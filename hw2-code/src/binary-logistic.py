# A6
# Binary logistic regression implementation

import matplotlib.pyplot as plt
import numpy as np

import constants as c
import helpers as h

# from scipy import

def main():
  print("binary logistic regression")

  X_train, labels_train, one_train, X_test, labels_test, one_test = h.load_mnist()

  a4 = GradientDescent(c.reg_lambda)

  a4.grad_desc(X_train, labels_train, c.mnist_step_size, c.mnist_offset)


# Implement gradient descent with an initial iterate of all zeros. Try several values of step sizes
# to find one that appears to make convergence on the training set as fast as possible. Run until
# you feel you are near to convergence.
class GradientDescent:
  def __init__(self, reg_lambda):
    self.reg_lambda = reg_lambda
    self.grad = None

  def grad_desc(self, X, Y, w, b):
    print("gradient descent")

    n = len(Y)

    offset = b + (X.T).dot(w)
    exponent = np.multiply(-Y, offset)
    mu = 1 / (1 + np.exp(exponent))

    coef = np.multiply(-Y, X.T)
    grad_w = (1 / n) * (coef.T).dot(1 - mu) + 2 * self.reg_lambda * w

    grad_b = (1 / n) * np.multiply(-Y, (1 - mu))


class StochasticGradientDescent:
  def __init__(self, reg_lambda, batch_size):
    self.reg_lambda = reg_lambda
    self.batch_size = batch_size
    self.grad = None

  def stoch_grad_desc(self, X, Y, w, b):
    print("stochastic gradient descent")



if __name__ == "__main__":
  main()
