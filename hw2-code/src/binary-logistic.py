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
    self.w = None
    self.b = 0

  def grad_desc(self, X, Y, cutoff):
    print("gradient descent")
    
    self.n, self.d = X.shape
    # 12223, 784
    self.w = np.zeros((self.d, 1))

    

    j_func, grad_w, grad_b = self.get_j(X, Y)

    print(j_func.shape)
    print(grad_w.shape)
    print(grad_b.shape)

    j_data = [j_func[0][0]]
    class_data = [self.get_error(X, Y)]

    print(j_data)
    print(class_data)

    # while np.max(np.abs(self.w)) > cutoff:
      # keep iterating

      # do we add or subtract here?
      # self.w = self.w + self.step * grad_w
      # self.b = self.b + self.step * grad_b

      # j_func, grad_w, grad_b = self.get_j(X, Y)

      # j_data.append(j_func)
      # class_data.append(self.get_error(X, Y))

    # then test on test data

  def get_j(self, X, Y):
    offset = self.b + (self.w.T).dot(X.T)
    exponent = np.multiply(-Y, offset)
    mu = 1 / (1 + np.exp(exponent))

    j_func = (1 / self.n) * np.sum(np.log(1 / mu)) + self.lamb * (self.w.T).dot(self.w)

    coef = np.multiply(-Y, X.T)
    grad_w = (1 / self.n) * coef.dot((1 - mu).T) + 2 * self.lamb * self.w

    grad_b = (1 / self.n) * -Y.dot((1 - mu).T)

    return j_func, grad_w, grad_b

  def get_error(self, X, Y):
    print("calculate gradient descent error")

    sign = np.sign(self.b + (self.w.T).dot(X.T))
    sign = sign[0]

    print(Y)
    print(sign)
    match_count = np.sum([sign[idx] == val for idx, val in enumerate(Y.T)])

    return 1 - (match_count / self.n)

  def plot_objective(self, train_j, test_j):
    print("plot gradient descent error")

    # c.results_path

  def plot_error(self, train_error, test_error):
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
