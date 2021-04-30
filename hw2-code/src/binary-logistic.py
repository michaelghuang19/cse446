# A6
# Binary logistic regression implementation

import matplotlib.pyplot as plt
import numpy as np

import constants as c
import helpers as h

# from scipy import

def main():
  print("binary logistic regression")

  # Y-values are in [-1, +1] format
  X_train, labels_train, Y_train, X_test, labels_test, Y_test = h.load_mnist()
  print(X_train.shape)
  print(Y_train.shape)

  assert (len(X_train) > 0 and len(X_test) > 0)
  
  a4 = GradientDescent(c.reg_lambda, c.mnist_step_size)
  a4.grad_desc(X_train, Y_train, c.cutoff)
  # a4.grad_desc(X_train, labels_train)

# Implement gradient descent with an initial iterate of all zeros. Try several values of step sizes
# to find one that appears to make convergence on the training set as fast as possible. Run until
# you feel you are near to convergence.
class GradientDescent:
  def __init__(self, reg_lambda, step_size):
    self.lamb = reg_lambda
    self.grad = None
    self.step = step_size
    self.b = 0

  def grad_desc(self, X, Y, cutoff):
    print("gradient descent")
    
    self.n, self.d = X.shape
    # 12223, 784
    w_init = np.zeros((self.d, 1))

    j_func, grad_w, grad_b = self.get_j(X, Y, w_init, self.b)

    print(j_func.shape)
    print(grad_w.shape)
    print(grad_b.shape)

  def get_j(self, X, Y, w, b):
    offset = b + w.T.dot(X.T)
    exponent = np.multiply(-Y, offset)
    mu = 1 / (1 + np.exp(exponent))

    j_func = (1 / self.n) * np.log(1 / mu) + self.lamb * (w.T).dot(w)

    coef = np.multiply(-Y, X.T)
    grad_w = (1 / self.n) * coef.dot((1 - mu).T) + 2 * self.lamb * w

    grad_b = (1 / self.n) * -Y.dot((1 - mu).T)

    return j_func, grad_w, grad_b

  def get_error(self, X, Y, w, b):
    print("calcualte gradient descent error")



  def plot_objective(self):
    print("plot gradient descent error")

    # c.results_path

  def plot_error(self):
    print("plot gradient descent error")


class StochasticGradientDescent:
  def __init__(self, reg_lambda, batch_size):
    self.reg_lambda = reg_lambda
    self.batch_size = batch_size
    self.grad = None

  def stoch_grad_desc(self, X, Y, w, b):
    print("stochastic gradient descent")



if __name__ == "__main__":
  main()
