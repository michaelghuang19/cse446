# A6
# Binary logistic regression implementation

import matplotlib.pyplot as plt
import numpy as np

import constants as c
import helpers as h

from scipy import linalg

def main():
  print("binary logistic regression")

  # Y-values are in [-1, +1] format
  X_train, labels_train, Y_train, X_test, labels_test, Y_test = h.load_mnist()

  assert (len(X_train) > 0 and len(X_test) > 0)
  
  # Gradient Descent

  gd = GradientDescent(c.reg_lambda, c.mnist_step_size)
  train_j, test_j, train_error, test_error = gd.grad_desc(X_train, Y_train, X_test, Y_test, c.cutoff)

  # print(train_j)

  h.plot_function("Loss over Time", "a6_bi", "Iterations", "Loss", train_j, test_j)
  h.plot_function("Error over Time", "a6_bii", "Iterations", "Error", train_error, test_error)

  # Stochastic Gradient Descent
  
  sgd1 = StochasticGradientDescent(c.reg_lambda, c.mnist_step_size, 1)
  sgd100 = StochasticGradientDescent(c.reg_lambda, c.mnist_step_size, 100)

  # perform sgd

  # get loss

  # get error

  # plot loss

  # plot error

class GradientDescent:
  def __init__(self, reg_lambda, step_size):
    self.lamb = reg_lambda
    self.step = step_size
    self.w = None
    self.b = 0
    self.d = 0

  def grad_desc(self, X_train, Y_train, X_test, Y_test, cutoff):
    print("gradient descent")
    
    # 12223, 784
    _, self.d = X_train.shape
    self.w = np.zeros((self.d, 1))

    train_j_data = []
    test_j_data = []
    train_class_data = []
    test_class_data = []

    # emulate do-while
    while True:

      train_j_func, grad_w, grad_b = self.get_j(X_train, Y_train)
      test_j_func, _, _ = self.get_j(X_test, Y_test)

      self.w = self.w - (self.step * grad_w)
      self.b = self.b - (self.step * grad_b)

      train_j_data.append(train_j_func[0][0])
      test_j_data.append(test_j_func[0][0])
      train_class_data.append(self.get_error(X_train, Y_train))
      test_class_data.append(self.get_error(X_test, Y_test))

      if np.max(np.abs(grad_w)) < cutoff:
        break

    return train_j_data, test_j_data, train_class_data, test_class_data

  def get_j(self, X, Y):
    n, d = X.shape
    Y = np.expand_dims(Y, axis=1)

    offset = self.b + (self.w.T).dot(X.T)
    exponent = np.multiply(-Y, offset.T)
    mu = 1 / (1 + np.exp(exponent))

    reg = self.lamb * (self.w.T).dot(self.w)
    j_func = (1 / n) * np.sum(np.log(1 / mu)) + reg

    coef = np.multiply(X, -Y)
    grad_reg = 2 * self.lamb * self.w
    grad_w = (1 / n) * (coef.T).dot(1 - mu) + grad_reg
    grad_w = grad_w.sum(axis = 1)
    grad_w = np.expand_dims(grad_w, axis=1)

    grad_b = (1 / n) * (-Y.T).dot(1 - mu)

    return j_func, grad_w, grad_b[0][0]

  def get_error(self, X, Y):
    n, d = X.shape

    sign = np.sign(self.b + (self.w.T).dot(X.T))
    sign = sign[0]

    match_count = np.sum([sign[idx] == val for idx, val in enumerate(Y.T)])

    return 1 - (match_count / n)

class StochasticGradientDescent:
  def __init__(self, reg_lambda, step_size, batch_size):
    self.lamb = reg_lambda
    self.step = step_size
    self.batch = batch_size
    self.w = None
    self.b = 0
    self.d = 0

  def stoch_grad_desc(self, X, Y, w, b):
    print("stochastic gradient descent")

  def get_j(self, X, Y):
    n, d = X.shape
    Y = np.expand_dims(Y, axis=1)

    offset = self.b + (self.w.T).dot(X.T)
    exponent = np.multiply(-Y, offset.T)
    mu = 1 / (1 + np.exp(exponent))

    reg = self.lamb * (self.w.T).dot(self.w)
    j_func = (1 / n) * np.sum(np.log(1 / mu)) + reg

    coef = np.multiply(X, -Y)
    grad_reg = 2 * self.lamb * self.w
    grad_w = (1 / n) * (coef.T).dot(1 - mu) + grad_reg
    grad_w = grad_w.sum(axis = 1)
    grad_w = np.expand_dims(grad_w, axis=1)

    grad_b = (1 / n) * (-Y.T).dot(1 - mu)

    return j_func, grad_w, grad_b[0][0]

  def get_error(self, X, Y):
    n, d = X.shape

    sign = np.sign(self.b + (self.w.T).dot(X.T))
    sign = sign[0]

    match_count = np.sum([sign[idx] == val for idx, val in enumerate(Y.T)])

    return 1 - (match_count / n)

if __name__ == "__main__":
  main()
