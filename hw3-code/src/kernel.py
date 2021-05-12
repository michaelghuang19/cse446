# kernel.py
# Applicable code for HW3 A3

import numpy as np

import constants as c
import helpers as h

def main():
  print("hello world")

  X, Y = h.generate_data()
  # shapes: (30, )

  # np.expand_dims(Y, axis=1)

  kf_poly = lambda x, z, d: (1 + (x.T).dot(z)) ** d
  kf_rbf = lambda x, z, gamma: np.exp(-gamma * (np.linalg.norm(x - z, 2) ** 2))

  k_poly = Kernel(X, Y, kf_poly)
  k_rbf = Kernel(X, Y, kf_rbf)

class Kernel:
  def __init__(self, X, Y, kernel_func=None, hyperparameter=None, lambda_reg=None):
    self.X = X
    self.Y = Y
    self.kf = kernel_func
    self.hp = hyperparameter
    self.lamb = lambda_reg
    self.kmat = None

    if self.hp is None:
      self.loo_cv()

  # use
  def loo_cv(self):
    print("finding best parameters for this function")

    n, d = self.X.shape

    error_matrix = np.zeros((len(c.hp_list), len(c.lamb_list)))

    # literally try every combo and keep track
    for i, hp in enumerate(c.hp_list):
      for j, lamb in enumerate(c.lamb_list):
        for k in range(n):
          X_val = self.X[k]
          Y_val = self.Y[k]
          
          # essentially remove the val
          X_train = np.concatenate((self.X[:k], self.X[k + 1:]))
          Y_train = np.concatenate((self.Y[:k], self.Y[k + 1:]))

          condensed_kf = lambda x, z: kf(x, z, hp)
          f_opt = kernel_rr(X_train, Y_train, condensed_kf, lamb)

          error_matrix[i][j] += np.mean((Y_train - f_opt(X_val)) ** 2)

    # find minimum index loc
    np.where(error_matrix == min(error_matrix))
    # return c.hp_list[min_i], c.lamb_list[min_j]
  
  def kernel_rr(self, X_train, Y_train, condensed_kf, lamb):
    n, d = self.X.shape
    # since we leave one out
    n = n - 1

    kmat = np.fromfunction(lambda i, j: self.kf(X_train[i], Y_train[j]))

    # use optimizer, train
    alpha_opt = np.linalg.solve(kmat + lamb * np.eye(n), Y_train)

    f_opt = lambda x: np.sum(alpha_opt.dot(alpha_opt, condensed_kf(x, X_train)))

    return f_opt

if __name__ == "__main__":
  main()
