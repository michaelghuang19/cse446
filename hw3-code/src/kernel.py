# kernel.py
# Applicable code for HW3 A3

import numpy as np

import constants as c
import helpers as h

def main():
  print("hello world")

  X, Y, true_f = h.generate_data()
  # shapes: (30, )

  """
  part (a)
  """
  
  # individual elements over entire matrix
  kf_poly = lambda x, z, d: (1 + x * z) ** d
  kf_rbf = lambda x, z, gamma: np.exp(-gamma * ((x - z) ** 2))

  k_poly = Kernel(X, Y, kf_poly)
  k_rbf = Kernel(X, Y, kf_rbf)

  """
  part (b)
  """

  true_data = [true_f(x_val) for x_val in c.x_list]
  poly_pred_data = k_poly.get_fhat_data()
  rbf_pred_data = k_rbf.get_fhat_data()

  poly_list = [true_data, poly_pred_data]
  rbf_list = [true_data, rbf_pred_data]

  h.plot_multiple("poly", "a5bi", X, Y, poly_list, c.pred_labels, c.a5b_ylimits)
  h.plot_multiple("rbf", "a5bii", X, Y, rbf_list, c.pred_labels, c.a5b_ylimits)

class Kernel:
  def __init__(self, X, Y, kernel_func=None, hyperparameter=None, lambda_reg=None):
    self.X = X
    self.Y = Y
    self.kf = kernel_func
    self.hp = hyperparameter
    self.lamb = lambda_reg
    self.kmat = None

    if self.hp is None and self.lamb is None:
      print("setting best hyperparameter, lambda")
      self.loo_cv()

    print("hyperparam: " + str(self.hp))
    print("lambda: " + str(self.lamb))

  def get_fhat_data(self):
    pred_f = self.kernel_rr(self.X, self.Y, self.hp, self.lamb)

    return [pred_f(x_val) for x_val in c.x_list]

  def loo_cv(self):
    n = len(self.X)

    error_matrix = np.zeros((len(c.hp_list), len(c.lamb_list)))

    # try every combo and keep track of errors
    for i, hp in enumerate(c.hp_list):
      for j, lamb in enumerate(c.lamb_list):
        for k in range(n):
          X_val = self.X[k]
          Y_val = self.Y[k]
          
          # train without the loo val
          X_train = np.concatenate((self.X[:k], self.X[k + 1:]))
          Y_train = np.concatenate((self.Y[:k], self.Y[k + 1:]))

          f_opt = self.kernel_rr(X_train, Y_train, hp, lamb)

          error_matrix[i][j] += (np.abs(f_opt(X_val) - Y_val) ** 2)
        
        error_matrix[i][j] /= n

    # find minimum index loc
    min_idx = np.argwhere(error_matrix == np.amin(error_matrix))
    min_i = min_idx[0][0]
    min_j = min_idx[0][1]

    assert (error_matrix[min_i][min_j] == np.amin(error_matrix)), "Sanity check min error is accurate"

    self.hp = c.hp_list[min_i]
    self.lamb = c.lamb_list[min_j]
  
  def kernel_rr(self, X_train, Y_train, hp, lamb):
    n = len(X_train)

    kmat = np.fromfunction(lambda i, j: self.kf(X_train[i], X_train[j], hp), shape=(n, n), dtype=int)

    # use optimizer, train
    alpha_opt = np.linalg.solve(kmat + lamb * np.eye(n), Y_train)

    pred_f = lambda x: alpha_opt.dot(self.kf(X_train, x, hp))

    return pred_f

if __name__ == "__main__":
  main()
